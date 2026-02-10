# Temple University Libraries Digital Collections

A curated IIIF exhibit showcasing 10 digital collections from Temple University Libraries, built with [Hugo](https://gohugo.io/).

## Collections

1. **Allied Posters of World War I** - WWI propaganda and recruitment posters
2. **John W. Mosley Photographs** - African American social life in Philadelphia (1930s-1960s)
3. **Philadelphia Dance Collection** - Pennsylvania Ballet and Philadanco photography
4. **Frank G. Zahn Railroad Photographs** - American railroading from steam to diesel
5. **City Parks Association Photographs** - Philadelphia parks and recreation (late 1800s-early 1900s)
6. **NAACP Philadelphia Branch Photographs** - Civil rights activities and community organizing
7. **Temple Sheet Music Collections** - 19th and early 20th century American sheet music
8. **Paul Robeson Collection** - Life and career of the singer, actor, and activist
9. **Blockson Afro-American Collection Photographs** - African American history and culture
10. **Temple History in Photographs** - Temple University from 1884 to present

## Features

- All images served via the IIIF Image API from Temple's ContentDM instance
- Static site built with Hugo for fast loading and easy hosting
- Responsive design for all screen sizes
- Each collection has a dedicated page with featured items
- Homepage provides an overview with preview images for each collection

## Local Development

```bash
hugo server -D
```

## Deployment

Automatically deployed to GitHub Pages via GitHub Actions on push to `main`.
