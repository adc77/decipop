# Article development

Read the published article at **https://adc77.github.io/decipop/**.

The page follows the typography and narrow dark layout of axondendrite.com. HTML,
CSS and JavaScript live in `site/`. No framework, analytics or external font is
required. Controls display saved results; they do not call a model.

To edit and preview locally:

```sh
python3 tools/build_article.py
python3 -m http.server 8765 --bind 127.0.0.1 --directory site
```

The data builder reads committed audit receipts under `reports/` and the recorded
Farah example under `examples/`. It rejects invalid raw-count receipts and embeds
source SHA-256 hashes in `site/assets/results.json`. Evidence links point to GitHub
rather than duplicated files inside the site. Use the print button for a PDF.

`.github/workflows/pages.yml` deploys the committed `site/` folder on updates to
`main`. Readers need only the public URL; the local server is for development.
