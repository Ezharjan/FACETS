# FACETS-PR · Project Page

Project page for the paper

> **Measuring Source Fidelity, Exculpatory Recall, and Generative Suspicion in AI-Drafted Police Reports**
> Aizierjiang Aiersilan (The George Washington University) and Yeershanati Wuernikebai (Washington and Lee University School of Law)
> *AAAI 2026 Fall Symposium Series*

The paper introduces FACETS-PR, a six-axis framework of sixteen metrics with a
vendor-independent protocol for measuring the substantive quality of AI-drafted
police reports, and instantiates it on four open-weight models over the
synthetic BWCSyn-90 body-worn camera corpus.

## Links

| Button  | Target | Status |
|---------|--------|--------|
| Paper   | <https://aaai.org/conference/fall-symposia/2026-fall-symposium-series-2/> | Placeholder (AAAI FSS 2026 main page); replace with the published paper URL |
| arXiv   | <https://arxiv.org/> | Placeholder (arXiv main page); replace with the arXiv abstract URL |
| Code    | <https://github.com/Ezharjan/FACETS> | Final |
| Dataset | <https://github.com/Ezharjan/FACETS/tree/master/data> | Final (BWCSyn-90 corpus) |
| Poster  | `paper_poster.pdf` | Hidden until the real poster is added |
| Confab  | <https://cal.com/ezhar/30min> | Final |

To show the poster, replace `paper_poster.pdf` with the paper's poster (an
unencrypted PDF), then set `"poster": { "enabled": true, ... }` and the Poster
link's `"enabled": true` in `config.json`.

## Editing the page

All page content (title, authors, links, abstract, tables, figures, BibTeX)
lives in `config.json`; `index.html` and `static/js/render.js` only render it.
`config.schema.json` provides autocomplete and validation in editors such as
VS Code. Figures are PNG files in `static/images/`.

## Previewing locally

The page loads `config.json` with `fetch()`, which browsers block for
`file://` URLs, so serve the folder instead:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Deploying (GitHub Pages)

1. Push this folder to a GitHub repository.
2. In **Settings → Pages**, choose **Deploy from a branch**, branch `main`
   (or `master`), folder `/ (root)`.
3. The page is served at `https://<user>.github.io/<repo>/`.

Any other static host works as well.

## Structure

```
index.html            Page shell (renders config.json)
config.json           All page content
config.schema.json    Schema for config.json
favicon.ico           Tab icon
paper_poster.pdf      Poster placeholder (hidden on the page)
static/css/           Bulma, Font Awesome and the page theme (index.css)
static/js/render.js   Renderer
static/images/        Figures
static/webfonts/      Font Awesome fonts
```

Layout adapted from the [Nerfies](https://github.com/nerfies/nerfies.github.io)
project page template.
