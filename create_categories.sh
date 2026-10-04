#!/bin/bash

# Directory containing the posts
POSTS_DIR="_posts"
# Directory containing the category pages
CATEGORY_DIR="category"

# Ensure category directory exists
mkdir -p "$CATEGORY_DIR"

# Find all unique categories from posts
# Look for lines starting with "categories:" or "category:" ONLY within the YAML front matter
# Extract all fields after the prefix, remove commas and brackets, and get unique values
CATEGORIES=$(awk '
    # If we are at the first line of a file and it is "---", start front matter block
    FNR==1 && /^---[ \t]*$/ { 
        in_fm = 1
        next 
    }
    
    # If we hit "---" while in front matter, end the block
    /^---[ \t]*$/ && in_fm { 
        in_fm = 0
        next 
    }
    
    # If we are inside the front matter and find category line, print the items
    in_fm && /^categor(y|ies):/ { 
        for (i = 2; i <= NF; i++) {
            print $i 
        }
    }
' "$POSTS_DIR"/*.md | tr -d '[],' | sort -u)

for cat in $CATEGORIES; do
    # Remove any stray carriage returns
    cat=$(echo "$cat" | tr -d '\r')
    
    # Skip empty lines
    if [ -z "$cat" ]; then
        continue
    fi
    
    FILE_PATH="$CATEGORY_DIR/${cat}.md"
    
    if [ ! -f "$FILE_PATH" ]; then
        # Create title case: replace '-' with space, capitalize first letter of each word
        TITLE=$(echo "$cat" | tr '-' ' ' | awk '{
            for (i = 1; i <= NF; i++) {
                $i = toupper(substr($i, 1, 1)) tolower(substr($i, 2))
            }
            print $0
        }')
        
        echo "Creating category page: $FILE_PATH (Title: $TITLE)"
        
        cat <<EOF > "$FILE_PATH"
---
layout: posts_by_category
title: $TITLE
permalink: /category/$cat
categories: $cat
---

EOF
    fi
done

echo "Finished creating missing category pages!"
