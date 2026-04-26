# Trivision

Trivision is a Next.js application for a text-to-3D workspace. The app keeps its current design system, uses Turso/libSQL for auth and workspace state, stores uploaded/generated files in Vercel Blob, and routes provider-backed 3D jobs through a modular generation architecture.

## Local Development

**Prerequisites:** Node.js 20+

1. Install dependencies with `npm install`.
2. Copy `.env.example` to `.env.local`.
3. Set `DATABASE_URL` and `DATABASE_AUTH_TOKEN` from Turso.
4. Set `BLOB_READ_WRITE_TOKEN` from Vercel Blob.
5. Set `RUNWARE_API_KEY` for the Runware-backed models.
6. Set `LIGHTNING_TRELLIS_API_URL` for the self-hosted Lightning TRELLIS.2 provider.
7. Start the development server with `npm run dev`.
8. Build for production with `npm run build`.
9. Run the production server with `npm run start`.

## Generation Architecture

- The shared model registry and parameter schema live under `lib/generation/registry.ts`.
- Provider adapters live under `lib/generation/providers/`, one file per model.
- Shared Runware transport logic lives in `lib/generation/runware-client.ts`.
- The self-hosted Lightning TRELLIS client lives in `lib/generation/lightning-client.ts`.
- Uploaded source images, masks, and generated 3D assets are stored through `lib/storage/blob.ts`.
- Generation jobs are stored in Turso/libSQL and exposed through `POST /api/generations` and `GET /api/generations/:id`.
- Completed source and output assets are streamed back through authenticated project asset routes.

## Current Model Support

- `microsoft:trellis-2@4b` is enabled for image-to-3D generation.
- `lightning:microsoft-trellis-2@4b` is enabled as a second TRELLIS.2 provider and runs `/rembg` before generation on the self-hosted Lightning API.
- `meta:sam@3d` is enabled for image-plus-mask 3D generation.

## Hosted Data Layer

- The app requires a Turso/libSQL `DATABASE_URL` and `DATABASE_AUTH_TOKEN`.
- Login, signup, dashboard, studio, workspaces, favorites, settings, and recovery flows all read from or write to this hosted database.
- New accounts create a clean workspace and default generation preferences. No demo projects or seeded accounts are injected automatically.
- Protected app routes require a valid HTTP-only session cookie; anonymous access redirects to `/login`.

## Vercel Deployment Notes

- Configure `DATABASE_URL` and `DATABASE_AUTH_TOKEN` from a Turso/libSQL database. The schema bootstrap is idempotent and no longer drops existing tables when the schema version changes.
- Configure `BLOB_READ_WRITE_TOKEN` from Vercel Blob so uploaded source images, masks, and generated outputs persist beyond a single function invocation.
- Configure `RUNWARE_API_KEY`, `RUNWARE_API_URL`, and `LIGHTNING_TRELLIS_API_URL` for provider-backed generation.
- Generation is processed inside `POST /api/generations`, which keeps the request lifecycle explicit and avoids in-memory background work.
- `POST /api/generations` and `POST /api/providers/lightning/rembg` run on the Node.js runtime with `maxDuration = 60`.

## Adding Providers or Models

- Add the model definition and parameter schema in `lib/generation/registry.ts`.
- Add a provider adapter in `lib/generation/providers/` and register it in `lib/generation/providers/index.ts`.
- Keep model-specific options inside the registry schema. The studio UI, validation, defaults, and request normalization all read from that schema.
