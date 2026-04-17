# Trivision

Trivision is a Next.js application for a text-to-3D workspace MVP. The app keeps its current design system, uses a local SQL database for auth and workspace state, stores uploaded/generated files on the local filesystem, and now routes provider-backed 3D jobs through a modular generation architecture.

## Local Development

**Prerequisites:** Node.js 20+

1. Install dependencies with `npm install`.
2. Copy `.env.example` to `.env.local`.
3. Set `RUNWARE_API_KEY` for the Runware-backed models.
4. Set `LIGHTNING_TRELLIS_API_URL` if you want to use the self-hosted Lightning TRELLIS.2 provider.
5. Start the development server with `npm run dev`.
6. Build for production with `npm run build`.
7. Run the production server with `npm run start`.

## Generation Architecture

- The shared model registry and parameter schema live under `lib/generation/registry.ts`.
- Provider adapters live under `lib/generation/providers/`, one file per model.
- Shared Runware transport logic lives in `lib/generation/runware-client.ts`.
- The self-hosted Lightning TRELLIS client lives in `lib/generation/lightning-client.ts`.
- Local uploads and generated 3D assets are stored under `data/storage/` through `lib/storage/local.ts`.
- Generation jobs are stored in the local database and exposed through `POST /api/generations` and `GET /api/generations/:id`.
- Completed source and output assets are streamed back through authenticated project asset routes.

## Current Model Support

- `microsoft:trellis-2@4b` is enabled for image-to-3D generation.
- `lightning:microsoft-trellis-2@4b` is enabled as a second TRELLIS.2 provider and runs `/rembg` before generation on the self-hosted Lightning API.
- `meta:sam@3d` is enabled for image-plus-mask 3D generation.

## Local Data Layer

- The app auto-creates a local database at `data/trivision.local.db`.
- Login, signup, dashboard, studio, workspaces, favorites, libraries, settings, and recovery flows all read from or write to this local store.
- The repository layer lives under `lib/db/` so future Supabase migration work can replace the persistence backend without rewriting the page components.
- Protected app routes require a valid local session cookie; anonymous access redirects to `/login`.

## Seeded Test Account

- Email: `technical.artist@trivision.io`
- Password: `Trivision123!`
- This account ships with demo workspace data. Newly created accounts get their own empty workspace and isolated generation history.
