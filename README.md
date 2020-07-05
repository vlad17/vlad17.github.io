# personal-web

Repository for local creation of my personal web site: [vlad17.github.io](vlad17.github.io).

## Frameworks and Infrastructure

I use [Jekyll](https://jekyllrb.com/) to generate my content and [GitHub Pages](https://pages.github.com/) to host it.

## Building and Deploying Locally

Install:

1. Transitive deps for Jekyll (not Jekyll itself): `nodejs`, `ruby`.
1. Make sure you have `bundler`: `sudo gem install bundler`
1. In the root directory, `bundler update && bundler install`

Local deployment:

1. In root directory, run `bundler exec jekyll serve`

## Credits

Site theme is from [this repository](https://github.com/codeasashu/hcz-jekyll-blog), moderately gutted.

## ipynb -> md page

```bash
bash mdconvert.sh assets/2020-07-04-complex-hash-collisions.ipynb
```
