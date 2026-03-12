# ============================================================
# 🔺 TriVision Studio — Colab Launch Script
#
# NON-BLOCKING: Starts the Flask server in a daemon thread,
# verifies health, sets up the Colab proxy URL, then RETURNS.
# The cell finishes executing, freeing the kernel to run
# the colab_keepalive.py cell (which is what actually prevents
# Colab from disconnecting).
#
# Usage (Colab cell):
#   import os
#   os.chdir("/content/TriVision/trivision-studio")
#   from server import *
#   exec(open("launch.py").read())
#
# Then in the NEXT cell, run:
#   exec(open("/content/TriVision/colab_keepalive.py").read())
# ============================================================

import os, time, threading, traceback, sys
import socket as _socket
import requests as _requests

# ── These come from server.py (already imported via `from server import *`) ──
# app, IN_COLAB, eval_js, jobs, console_lines


def _find_free_port(preferred=5000):
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", preferred))
            return preferred
        except OSError:
            s.bind(("0.0.0.0", 0))
            return s.getsockname()[1]


PORT = _find_free_port(5000)
os.environ["TRIVISION_PORT"] = str(PORT)
_server_error = [None]


def _run_server():
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
    except Exception as exc:
        _server_error[0] = exc
        sys.__stderr__.write(f"\n❌ Flask server crashed: {exc}\n")
        traceback.print_exc(file=sys.__stderr__)


server_thread = threading.Thread(target=_run_server, daemon=True, name='flask-server')
server_thread.start()

# ══════════════════════════════════════════════════════════════
# 1) Wait until Flask is responding on localhost
# ══════════════════════════════════════════════════════════════

_local_ready = False
_last_local_err = None
for _attempt in range(80):
    if _server_error[0] is not None:
        break
    try:
        _r = _requests.get(f"http://127.0.0.1:{PORT}/api/keepalive", timeout=0.75)
        if _r.status_code == 200:
            _local_ready = True
            break
    except Exception as _e:
        _last_local_err = _e
    time.sleep(0.5)

if not _local_ready:
    raise RuntimeError(
        f"Flask server never became reachable on localhost:{PORT}.\n"
        f"  Server-thread error : {_server_error[0]}\n"
        f"  Last health-check error: {_last_local_err}\n"
    )

print(f"✅ Flask server healthy on localhost:{PORT}")

# ══════════════════════════════════════════════════════════════
# 2) Obtain & verify public access (Colab only)
# ══════════════════════════════════════════════════════════════

_launch_mode = None
if IN_COLAB:
    from IPython.display import display, HTML as _HTML

    _POPOUT_WIDGET = """
    <div style="margin:8px 0 4px;padding:10px 16px;background:#141414;border:2px solid #E8A917;border-radius:12px;font-family:monospace;">
        <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;">
            <span style="color:#E8A917;font-weight:bold;">🔺 TriVision Studio</span>
            <span style="color:#8A8A8A;font-size:13px;">embedded above and ready for pop-out</span>
            <button id="trivision-popout-btn" style="
                padding:6px 14px;background:#E8A917;color:#141414;border:none;border-radius:6px;
                font-family:monospace;font-size:13px;font-weight:bold;cursor:pointer;
            ">↗ Open in new tab</button>
        </div>
        <div id="trivision-popout-status" style="margin-top:8px;color:#8A8A8A;font-size:12px;">
            Browser-side proxy resolution is used here because static Colab proxy links can expire or fail.
        </div>
    </div>
    <script>
    (function() {
        const btn = document.getElementById('trivision-popout-btn');
        const status = document.getElementById('trivision-popout-status');
        if (!btn || !status) return;

        async function resolveLaunchTarget() {
            const nativeLink = Array.from(document.querySelectorAll('a')).find(
                anchor =>
                    anchor.textContent &&
                    anchor.textContent.includes('TriVision Studio') &&
                    anchor.href
            );
            if (nativeLink && nativeLink.href) {
                return {kind: 'native-link', url: nativeLink.href};
            }
            const iframe = Array.from(document.querySelectorAll('iframe')).find(
                frame => frame.src && frame.src.includes('prod.colab.dev')
            );
            if (iframe && iframe.src) {
                return {kind: 'iframe-src', url: iframe.src};
            }
            let url = await google.colab.kernel.proxyPort(%d, {cache: false});
            if (url && !url.startsWith('http')) url = 'https://' + url;
            if (url && !url.endsWith('/')) url += '/';
            return {kind: 'proxy-port', url};
        }

        btn.addEventListener('click', async function() {
            btn.disabled = true;
            btn.textContent = 'Opening...';
            status.textContent = 'Resolving the best Colab launch link...';
            try {
                const target = await resolveLaunchTarget();
                const url = target.url;
                if (target.kind !== 'native-link') {
                    const health = url.replace(/\/$/, '') + '/api/keepalive?ts=' + Date.now();
                    try {
                        await fetch(health, {credentials: 'include', cache: 'no-store'});
                    } catch (_) {}
                }

                const win = window.open(url, '_blank', 'noopener,noreferrer');
                if (!win) {
                    if (target.kind === 'native-link') {
                        const nativeLink = Array.from(document.querySelectorAll('a')).find(
                            anchor =>
                                anchor.textContent &&
                                anchor.textContent.includes('TriVision Studio') &&
                                anchor.href === url
                        );
                        if (nativeLink) {
                            nativeLink.click();
                        }
                    } else {
                        const a = document.createElement('a');
                        a.href = url;
                        a.target = '_blank';
                        a.rel = 'noopener noreferrer';
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                    }
                }

                const labels = {
                    'native-link': 'Colab window link',
                    'iframe-src': 'embedded iframe URL',
                    'proxy-port': 'proxy URL fallback'
                };
                status.innerHTML = 'Opened via ' + labels[target.kind] + ': <a href="' + url + '" target="_blank" rel="noopener noreferrer" style="color:#E8A917;text-decoration:underline;">' + url + '</a>';
                btn.textContent = '↗ Open in new tab';
            } catch (e) {
                status.textContent = 'Open failed: ' + (e && e.message ? e.message : e);
                btn.textContent = '⚠ Try again';
            } finally {
                btn.disabled = false;
            }
        });
    })();
    </script>
    """ % PORT

    _iframe_ok = False
    _window_ok = False
    try:
        from google.colab import output as _colab_output
        _colab_output.serve_kernel_port_as_iframe(PORT, path='/', height='820')
        _launch_mode = "iframe"
        _iframe_ok = True
    except Exception as _iframe_err:
        sys.__stdout__.write(f"  ⚠ iframe fallback failed: {_iframe_err}\n")

    try:
        from google.colab import output as _colab_output
        _colab_output.serve_kernel_port_as_window(PORT, path='/', anchor_text="🔺 Click to open TriVision Studio")
        _window_ok = True
        if not _iframe_ok:
            _launch_mode = "window"
    except Exception as _window_err:
        sys.__stdout__.write(f"  ⚠ window helper failed: {_window_err}\n")
        if not _iframe_ok:
            try:
                from IPython.display import Javascript as _JS
                display(_JS("""
                (async () => {
                    const url = await google.colab.kernel.proxyPort(%d, {cache: false});
                    const iframe = document.createElement('iframe');
                    iframe.src = url;
                    iframe.width = '100%%';
                    iframe.height = '820';
                    iframe.style.border = '2px solid #E8A917';
                    iframe.style.borderRadius = '12px';
                    document.querySelector('#output-area').appendChild(iframe);
                })();
                """ % PORT))
                _launch_mode = "js_iframe"
            except Exception:
                raise RuntimeError(
                    f"All Colab display methods failed for port {PORT}.\n"
                    f"  The Flask server IS running on localhost:{PORT}.\n"
                )

    if _window_ok:
        display(_HTML("""
        <div style="margin:8px 0;padding:10px 16px;background:#141414;border:2px solid #E8A917;border-radius:12px;font-family:monospace;">
            <span style="color:#8A8A8A;font-size:13px;">If the native Colab link errors, use the button below instead.</span>
        </div>
        """))
    display(_HTML(_POPOUT_WIDGET))

    print(f"🚀 Launch mode: {_launch_mode}")
else:
    _launch_mode = "local"
    print(f"\n🔺 TriVision Studio running at http://localhost:{PORT}\n")

# ══════════════════════════════════════════════════════════════
# DONE — Cell finishes here. Server continues in background.
# Now run colab_keepalive.py in the NEXT cell.
# ══════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("  ✅ Server is running in background thread.")
print("  👉 Run the NEXT cell (colab_keepalive.py) to prevent disconnect.")
print("=" * 60)
