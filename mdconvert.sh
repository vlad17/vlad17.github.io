#!/bin/bash
#
# Used to convert a jupyter notebook into jekyll markdown
# this is hard and annoying b/c jekyll expects some kind
# of insane latex escaping

if [ $# -ne 1 ] ; then
    echo "Usage: bash mdconvert.sh assets/blog-nb/some-jupyter-notebook.ipynb" >&2
fi

pushd assets/blog-nb
nbfile="$1"
poetry run jupyter nbconvert --to markdown $(basename "$nbfile")
popd
mv ${nbfile%.ipynb}.md _posts/
mdfile=_posts/$(basename ${nbfile%.ipynb}.md)
mv $mdfile ${mdfile}-old

# centered equations get their own lines, markdown
# latex is a nightmare

sed 's/\$\$/\n\n\$\$\n\n/g' ${mdfile}-old | cat -s | \
  awk '
BEGIN { inside_equation = 0 }
{
  if ($0 ~ /^\$\$$/) { 
    if (inside_equation) print "\\\\]"
    else print "\\\\["
    inside_equation = 1 - inside_equation
  }
  else {
    if (inside_equation) {
      gsub(/\\[{]/, "\\\\\\\\{");
      gsub(/\\[}]/, "\\\\\\\\}");
      gsub(/_/, "\\_");
      gsub(/[|]/, "\\|");
    }
    if (NF || !inside_equation) print $0
  }
}' | \
  sed -e 's/\$\([^$]*\)\$/\n\\\\(\n\1\n\\\\)\n/g' | \
  awk '
BEGIN { inside_equation = 0 }
{
  if ($0 ~ /^\\\\[(]$/) inside_equation = 1;
  if (inside_equation) {
    gsub(/_/, "\\_");
    gsub(/\\[{]/, "\\\\\\\\{");
    gsub(/\\[}]/, "\\\\\\\\}");
    gsub(/[|]/, "\\|");
  }
  print $0
  if ($0 ~ /^\\\\[)]$/) inside_equation = 0;
}' | \
  sed '/^\\\\[()]/N;s/\n//' | \
  sed -e :a -e '$!N;s/\n\\\\\([()]\)/ \\\\\1/;ta' -e 'P;D' | \
  sed 's/!\[png\](/![png](\/assets\/blog-nb\//g' > $mdfile
mv $mdfile ${mdfile}-old
dt=$(basename $mdfile | cut -c1-10)
echo $dt
bn=$(basename $mdfile)
bn=$(echo $bn | cut -c12-)
bn=${bn%.md}
bn=${bn//-/ }
arr=( $bn )
bn="${arr[@]^}"
echo "---
layout: post
title:  ${bn}
date:   ${dt}
categories: tools
---" > ${mdfile}
cat ${mdfile}-old >> ${mdfile}
echo "

[Try the notebook out yourself.](/${nbfile})
" >> ${mdfile}
rm ${mdfile}-old
