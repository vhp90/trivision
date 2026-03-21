# Trivision

Trivision is a Next.js application for a text-to-3D workspace MVP. The app keeps its current design system, uses a local SQL database for auth and workspace state, stores uploaded/generated files on the local filesystem, and now routes provider-backed 3D jobs through a modular generation architecture.

## Local Development

**Prerequisites:** Node.js 20+

1. Install dependencies with `npm install`.
2. Copy `.env.example` to `.env.local`.
3. Set `RUNWARE_API_KEY` so the active TRELLIS.2 provider can execute real generations.
4. Start the development server with `npm run dev`.
5. Build for production with `npm run build`.
6. Run the production server with `npm run start`.

## Generation Architecture

- The shared model registry and parameter schema live under `lib/generation/registry.ts`.
- Provider adapters live under `lib/generation/providers/`, one file per model.
- Shared Runware transport logic lives in `lib/generation/runware-client.ts`.
- Local uploads and generated 3D assets are stored under `data/storage/` through `lib/storage/local.ts`.
- Generation jobs are stored in the local database and exposed through `POST /api/generations` and `GET /api/generations/:id`.
- Completed source and output assets are streamed back through authenticated project asset routes.

## Current Model Support

- `microsoft:trellis-2@4b` is enabled for image-to-3D generation.
- `meta:sam@3d` is implemented in the registry and provider layer but intentionally disabled in the UI until mask upload support is added.

## Local Data Layer

- The app auto-creates a local database at `data/trivision.local.db`.
- Login, signup, dashboard, studio, workspaces, favorites, libraries, settings, and recovery flows all read from or write to this local store.
- The repository layer lives under `lib/db/` so future Supabase migration work can replace the persistence backend without rewriting the page components.
- Protected app routes require a valid local session cookie; anonymous access redirects to `/login`.

## Seeded Test Account

- Email: `technical.artist@trivision.io`
- Password: `Trivision123!`
- This account ships with demo workspace data. Newly created accounts get their own empty workspace and isolated generation history.
