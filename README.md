# personal-web

Repository for local creation of my personal web site:

TODO: aws link

## Frameworks and Infrastructure

I use [Jekyll](https://jekyllrb.com/) to generate my content and [s3_website](https://github.com/laurilehmijoki/s3_website) for deployment to Amazon S3.

TODO: deploy to s3
TODO: host with cloudfront

## Building and Deploying Locally

Install:

1. Transitive deps for Jekyll (not Jekyll itself).
1. Package binaries: `nodejs`
1. Make sure you have `bundler`: `sudo gem install bundler`
1. In `jekyll-project` subdirectory, `bundler update && bundler install`

Local deployment:

1. In `jekyll-project` subdirectory, run `bundler exec jekyll serve`

## Requirements for deploying on S3

1. Transitive dependencies for s3_website.


