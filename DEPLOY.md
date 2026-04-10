# RobBot Deployment

The fastest way to get RobBot on a public domain is to deploy it to Render.

## What this gives you

- A public HTTPS URL like `https://robbot.onrender.com`
- No need to keep your laptop running
- Easy environment-variable setup for the OpenAI key

## Before you deploy

Your local `.env` currently contains a real OpenAI API key. Rotate that key in the OpenAI dashboard before sharing this project or pushing it to any remote repository.

## Deploy on Render

1. Put the `RobBot` folder in a GitHub repo.
2. Log in to Render and choose `New +` -> `Blueprint`.
3. Point Render at the repo that contains this project.
4. Render will detect [`render.yaml`](./render.yaml) and create the web service.
5. When prompted for env vars, set `OPENAI_API_KEY`.
6. Deploy.

## App details

- Root directory: `RobBot`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn kb_service:app --host 0.0.0.0 --port $PORT`

## Notes

- The checked-in `knowledge_base.jsonl` and `chroma_db/` let the app answer questions without rebuilding the index during deploy.
- If you later add or change PDFs in `toIngest/`, rebuild locally with `python run_robbot.py` and redeploy.
- Free hosting tiers can spin down after inactivity, so the first load may take a short moment.
