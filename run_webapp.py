import os

from webapp import create_app

app = create_app()


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=app.config.get("DEBUG", False), host=host, port=port)
