# Trellis 2 API Reference

Base URL: `http://<host>:8000/api`

All endpoints are prefixed with `/api`. The server uses an async job queue — you submit a generation request, receive a job ID, poll for status, then download the result.

---

## Workflow Overview

```
1. POST /api/generate             →  get job_id
2. GET  /api/job/{job_id}/status  →  poll until status == "complete"
3. GET  /api/job/{job_id}/result  →  download .glb binary
```

---

## Endpoints

### POST `/api/generate`

Submit an image for 3D model generation. Accepts either **multipart/form-data** (recommended for file uploads) or **application/json** (base64-encoded image).

**Returns** `202 Accepted` on success with a job ID for polling.

#### Option A: Multipart Form Data

| Field               | Type      | Required | Default         | Description                                                        |
|---------------------|-----------|----------|-----------------|--------------------------------------------------------------------|
| `file`              | file      | *        | —               | Image file upload (JPEG, PNG, or WebP)                             |
| `image`             | string    | *        | —               | Base64-encoded image (alternative to `file`)                       |
| `seed`              | integer   | No       | `42`            | Random seed for reproducibility                                    |
| `pipeline_type`     | string    | No       | `"1024_cascade"`| Pipeline variant (see below)                                      |
| `preprocess_image`  | boolean   | No       | `true`          | Whether to run background removal / preprocessing on the image     |
| `decimation_target` | integer   | No       | `1000000`       | Target face count for mesh decimation                              |
| `texture_size`      | integer   | No       | `4096`          | Output texture resolution in pixels                                |
| `remesh`            | boolean   | No       | `true`          | Whether to remesh the output geometry                              |
| `simplify_limit`    | integer   | No       | `16777216`      | Max face count before simplification (nvdiffrast limit)            |

> \* You must provide either `file` OR `image`, not both.

#### Pipeline Types

| Value            | VRAM Required | Description                                                    |
|------------------|---------------|----------------------------------------------------------------|
| `"512"`          | ~16 GB        | Single-pass 512-resolution pipeline. Faster, lower VRAM usage  |
| `"1024_cascade"` | ~22+ GB       | Two-pass cascade: 512 then upscaled to 1024. Higher quality but requires more VRAM |

> If you're running on a GPU with less than 24 GB VRAM, use `"512"` to avoid CUDA out-of-memory errors.

#### Option B: JSON Body

Send `Content-Type: application/json` with the same fields. The `image` field must be a base64-encoded string.

```json
{
  "image": "<base64-encoded-image>",
  "seed": 42,
  "pipeline_type": "512",
  "preprocess_image": true,
  "decimation_target": 1000000,
  "texture_size": 4096,
  "remesh": true,
  "simplify_limit": 16777216
}
```

#### Success Response — `202 Accepted`

```json
{
  "job_id": "a1b2c3d4e5f6...",
  "queue_position": 1,
  "status": "queued"
}
```

| Field            | Type    | Description                                  |
|------------------|---------|----------------------------------------------|
| `job_id`         | string  | Unique job identifier — use this for polling  |
| `queue_position` | integer | 1-based position in the queue (0 = not queued)|
| `status`         | string  | Always `"queued"` on initial submission        |

#### Error Responses

| Status | Condition                        | Body                                                                          |
|--------|----------------------------------|-------------------------------------------------------------------------------|
| `400`  | Missing image (form-data)        | `{"error": "No image provided. Send a 'file' upload or 'image' base64 field.", "request_id": "..."}` |
| `400`  | Missing image (JSON)             | `{"error": "Missing 'image' field", "request_id": "..."}`                     |
| `400`  | Empty image data                 | `{"error": "Empty image data", "request_id": "..."}`                          |
| `400`  | Invalid base64 (form-data)       | `{"error": "Invalid base64 encoding in 'image' field", "request_id": "..."}`  |
| `400`  | Invalid base64 (JSON)            | `{"error": "Invalid base64 encoding", "request_id": "..."}`                   |
| `400`  | Corrupt / unsupported image      | `{"error": "Cannot decode image: ...", "request_id": "..."}`                  |
| `400`  | Invalid JSON body                | `{"error": "Invalid JSON body", "request_id": "..."}`                         |
| `429`  | Queue is full                    | `{"error": "Queue full", "request_id": "..."}`                                |

Supported image formats: **JPEG**, **PNG**, **WebP**.

---

### GET `/api/job/{job_id}/status`

Poll the current status of a generation job.

#### Path Parameters

| Parameter | Type   | Description          |
|-----------|--------|----------------------|
| `job_id`  | string | The job ID from `/generate` |

#### Success Response — `200 OK`

```json
{
  "job_id": "a1b2c3d4e5f6...",
  "status": "postprocessing",
  "elapsed_time": 45.123,
  "stage_times": {
    "preprocessing": 20.75,
    "sparse_structure": 0.001,
    "shape_generation": 0.001,
    "texture_generation": 0.001
  }
}
```

| Field          | Type              | Description                                                |
|----------------|-------------------|------------------------------------------------------------|
| `job_id`       | string            | The job identifier                                         |
| `status`       | string            | Current pipeline stage (see status values below)           |
| `elapsed_time` | float \| null     | Seconds since processing started. `null` if still queued   |
| `stage_times`  | object \| null    | Per-stage durations in seconds. `null` if no stages done   |

> **Note on stage_times:** The Trellis 2 pipeline runs sparse structure sampling, shape generation, and texture generation internally within a single `pipeline.run()` call. As a result, the bulk of the compute time is reported under `preprocessing`. The `sparse_structure`, `shape_generation`, and `texture_generation` stage times will appear as near-zero values. The `postprocessing` stage (mesh simplification, decimation, GLB export) is tracked separately and typically takes 60–100 seconds.

#### Job Status Values

| Status                | Description                                      |
|-----------------------|--------------------------------------------------|
| `queued`              | Waiting in queue, not yet started                 |
| `preprocessing`       | Running the generation pipeline (image preprocessing, sparse structure, shape, and texture sampling) |
| `sparse_structure`    | Sparse structure stage marker (see note above)    |
| `shape_generation`    | Shape generation stage marker (see note above)    |
| `texture_generation`  | Texture generation stage marker (see note above)  |
| `postprocessing`      | Mesh simplification, decimation, GLB export       |
| `complete`            | Done — result is ready for download               |
| `failed`              | Generation failed (check error on result endpoint)|
| `cancelled`           | Job was cancelled (server shutdown)               |

#### Error Responses

| Status | Condition       | Body                                      |
|--------|-----------------|-------------------------------------------|
| `404`  | Unknown job ID  | `{"error": "Job 'xxx' not found"}`        |

---

### GET `/api/job/{job_id}`

Alias for `/api/job/{job_id}/status`. Returns the same response.

---

### GET `/api/job/{job_id}/result`

Download the generated 3D model as a GLB binary file. The GLB uses WebP-compressed textures.

#### Path Parameters

| Parameter | Type   | Description          |
|-----------|--------|----------------------|
| `job_id`  | string | The job ID from `/generate` |

#### Success Response — `200 OK`

- **Content-Type:** `model/gltf-binary`
- **Content-Disposition:** `attachment; filename="{job_id}.glb"`
- **Body:** Raw GLB binary data (typically 30–60 MB with WebP textures)

#### Error Responses

| Status | Condition                          | Body                                                          |
|--------|------------------------------------|---------------------------------------------------------------|
| `400`  | Job failed                         | `{"error": "Job 'xxx' failed: <error detail>"}`               |
| `400`  | Job not yet complete               | `{"error": "Job 'xxx' is not yet complete (status: ...)"}`    |
| `404`  | Unknown job ID                     | `{"error": "Job 'xxx' not found"}`                            |
| `410`  | Result expired (TTL eviction)      | `{"error": "Result for job 'xxx' has expired"}`               |

> Results are retained for **10 minutes** (600s) after completion by default. Download promptly.

---

### GET `/api/health`

Server readiness and resource check.

#### Success Response — `200 OK`

```json
{
  "status": "ready",
  "uptime": 342.56,
  "gpu_memory_used_mb": 16443.86,
  "gpu_memory_total_mb": 22563.12
}
```

| Field                | Type   | Description                                |
|----------------------|--------|--------------------------------------------|
| `status`             | string | `"ready"` or `"loading"`                   |
| `uptime`             | float  | Server uptime in seconds                   |
| `gpu_memory_used_mb` | float  | Current GPU memory usage in MB             |
| `gpu_memory_total_mb`| float  | Total GPU memory available in MB           |

#### Loading Response — `503 Service Unavailable`

Same body shape but `status` is `"loading"`. Returned while models are still being loaded into GPU memory (typically 2–5 minutes on first startup).

---

## Global Error Response

During server shutdown, all endpoints return:

```
503 Service Unavailable
```
```json
{
  "error": "Server is shutting down",
  "status": "shutting_down"
}
```

---

## Frontend Integration Examples

### JavaScript — Full Generation Flow

```js
// 1. Submit image for generation
async function submitGeneration(imageFile, options = {}) {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('seed', options.seed ?? 42);
  formData.append('pipeline_type', options.pipelineType ?? '512');
  formData.append('preprocess_image', options.preprocessImage ?? true);
  formData.append('texture_size', options.textureSize ?? 4096);

  const res = await fetch('/api/generate', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error);
  }

  return res.json(); // { job_id, queue_position, status }
}

// 2. Poll for status
async function pollStatus(jobId, onProgress, intervalMs = 2000) {
  while (true) {
    const res = await fetch(`/api/job/${jobId}/status`);
    const data = await res.json();

    onProgress?.(data); // update UI with status/elapsed_time/stage_times

    if (data.status === 'complete') return data;
    if (data.status === 'failed' || data.status === 'cancelled') {
      throw new Error(`Job ${data.status}`);
    }

    await new Promise(r => setTimeout(r, intervalMs));
  }
}

// 3. Download GLB result
async function downloadResult(jobId) {
  const res = await fetch(`/api/job/${jobId}/result`);

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error);
  }

  return res.blob(); // GLB binary — use with three.js, model-viewer, etc.
}

// Usage
const { job_id } = await submitGeneration(file, { seed: 123 });
await pollStatus(job_id, (s) => console.log(s.status, s.elapsed_time));
const glbBlob = await downloadResult(job_id);
```

### Display GLB in Browser

```html
<!-- Using Google's <model-viewer> web component -->
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>

<model-viewer id="viewer" auto-rotate camera-controls></model-viewer>

<script>
  const blob = await downloadResult(jobId);
  const url = URL.createObjectURL(blob);
  document.getElementById('viewer').src = url;
</script>
```

### cURL Examples

```bash
# Submit a generation job
curl -X POST http://localhost:8000/api/generate \
  -F "file=@my_image.png" \
  -F "seed=42" \
  -F "pipeline_type=512"

# Poll status
curl http://localhost:8000/api/job/{job_id}/status

# Download result
curl -o output.glb http://localhost:8000/api/job/{job_id}/result

# Health check
curl http://localhost:8000/api/health
```

---

## Rate Limiting & Queue Behavior

- The server processes **one job at a time** (single GPU worker).
- Default max queue size: **10 jobs**. Submissions beyond this return `429`.
- `queue_position` in the submit response tells you where you are in line.
- Results expire after **10 minutes** — download the GLB before then or you'll get `410 Gone`.
- A typical generation takes **2–3 minutes** total (20s preprocessing + sampling, 60–100s postprocessing).

## CORS

By default the server allows all origins (`*`). This is configurable server-side via the `CORS_ORIGINS` environment variable (comma-separated list of allowed origins).

## Environment Variables

| Variable                    | Default                | Description                                    |
|-----------------------------|------------------------|------------------------------------------------|
| `HOST`                      | `0.0.0.0`             | Server bind address                            |
| `PORT`                      | `8000`                | Server port                                    |
| `CORS_ORIGINS`              | `*`                   | Comma-separated allowed origins                |
| `MAX_QUEUE_SIZE`            | `10`                  | Maximum concurrent jobs in queue               |
| `RESULT_RETENTION_SECONDS`  | `600`                 | How long completed results are kept (seconds)  |
| `MODEL_PATH`                | `Aero-Ex/Trellis2-GGUF` | Path to model weights directory             |
| `LOW_VRAM`                  | `false`               | Enable low-VRAM mode (lazy model loading)      |
| `ATTN_BACKEND`              | `sdpa`                | Attention backend (`sdpa`, `flash_attn`, `xformers`) |
