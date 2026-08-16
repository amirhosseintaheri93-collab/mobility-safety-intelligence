# OpenAI Setup

The dashboard is designed so normal users do not paste API keys.

Configure the key once on the server:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "replace-with-your-server-side-key"
OPENAI_MODEL = "gpt-5.2"
```

After this, users can select **Policy Agent -> OpenAI synthesis** and the app will call OpenAI automatically.

Do not commit `.streamlit/secrets.toml`. It is already ignored by `.gitignore`.
