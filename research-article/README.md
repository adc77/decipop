# Local research article

A local article about completed V1–V8 research, styled after the owner's posts at https://axondendrite.com: a narrow 36rem column, near-black background, system sans-serif type, restrained headings, muted text, and direct prose. Interactive figures use saved audited counts and preselected examples; they perform no model inference.

From the repository root:

```sh
python3 research-article/build.py
python3 -m http.server 8765 --bind 127.0.0.1 --directory research-article/dist
```

Open http://localhost:8765. The server exposes only the article and its curated evidence copies. The build verifies the independent raw-count receipts and copies source records with a SHA-256 manifest. No dependencies, analytics, external fonts, or network model calls are required.

Use the page's Print / PDF button for a print layout. A separate presentation manifest records this article; historical study archives remain unchanged. Code, model visibility, and Git publication are separate from this local preview.
