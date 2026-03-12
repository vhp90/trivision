import os
import time

from IPython import get_ipython
from IPython.display import HTML, display


def run_keepalive(interval_s=20):
    """Keep a Colab session active while the TriVision server is running."""
    try:
        import requests as _req

        port = int(os.environ.get("TRIVISION_PORT", "5000"))
        resp = _req.get(f"http://127.0.0.1:{port}/api/keepalive", timeout=2)
        if resp.status_code == 200:
            print(f"✅ TriVision server alive on port {port}")
        else:
            print(f"⚠ TriVision server returned {resp.status_code} on port {port}")
    except Exception:
        print("⚠ TriVision server not detected — run launch.py first")

    eval_js = None
    try:
        from google.colab.output import eval_js as _eval_js

        eval_js = _eval_js
        print("✅ Colab eval_js available")
    except ImportError:
        print("⚠ Not running inside Colab — keepalive helper is limited")

    display(HTML("""
    <script>
    (function() {
        function dismissDialogs() {
            const selectors = [
                'paper-button#ok',
                'mwc-button[slot="primaryAction"]',
                'md-text-button[slot="primaryAction"]',
                'md-filled-button[slot="primaryAction"]',
                'colab-dialog paper-button',
                'colab-dialog mwc-button'
            ];
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                if (el && el.offsetParent !== null) {
                    el.click();
                    return true;
                }
            }
            return false;
        }

        new MutationObserver(() => dismissDialogs()).observe(document.body, {
            childList: true,
            subtree: true
        });
        dismissDialogs();
        window.__trivision_keepalive = true;
    })();
    </script>
    """))

    shell = get_ipython()
    if shell is None:
        print("⚠ No active IPython shell found")
        return

    started = time.time()
    cycles = 0
    print(f"🔄 TriVision keepalive running every {interval_s}s. Interrupt the cell to stop.")

    try:
        while True:
            cycles += 1
            elapsed = int(time.time() - started)
            hh, rem = divmod(elapsed, 3600)
            mm, ss = divmod(rem, 60)
            shell.run_cell(
                f"import gc; gc.collect()  # TriVision keepalive {cycles} @ {hh:02d}:{mm:02d}:{ss:02d}",
                silent=False,
                store_history=False,
            )

            if eval_js is not None:
                try:
                    eval_js("""
                    (function() {
                        document.body.click();
                        const connect =
                            document.querySelector('colab-connect-button') ||
                            document.querySelector('#connect');
                        if (connect) connect.click();
                    })();
                    """, ignore_result=True)
                except Exception:
                    pass

            time.sleep(interval_s)
    except KeyboardInterrupt:
        print(f"\n🛑 TriVision keepalive stopped after {cycles} cycles.")
