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

Favicon image is from [this random site](http://www.playbuzz.com/martinshaba10/what-planet-describes-you-most).

## Other site TODOs:

* Dedup parallel -> only pram
* Spelling check (make it a script).
* Add images to project cards
* Make scripts (excluded from site) for generating new paper posts, new categories
* Add Apache licenses to all personal code
* Work info
* Edu info
* Fix the off-center sharing stuff
