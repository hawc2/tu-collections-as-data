# Design Document: Collections as Data Showcase

## 1. Goal
Transition the site from being a simple image gallery to a "Collections as Data" showcase. We will use a static data approach to model how digital collections can be analyzed computationally (e.g., tracking material volume, temporal range, and geographic spread). This will serve as a proof-of-concept for exploring IIIF collections as datasets.

## 2. Architecture & Data Strategy
- **Data Source**: We will continue using `data/collections.json` as the primary data store.
- **Data Enrichment**: We will enrich the JSON structure for each collection to include aggregate statistics (e.g., `totalItems`, `dateRange`, `topSubjects`) and structured visualization data (e.g., temporal distribution arrays or geographic coordinates).
- **Static Generation**: Hugo will parse these data structures and inject them into the HTML client-side scripts.
- **Visualization Tooling**: We will use lightweight, unopinionated client-side libraries to render the data.
  - **Timelines/Charts**: Chart.js (clean, easy to integrate, responsive).
  - **Maps**: Leaflet (lightweight, standard for basic mapping).

## 3. Visuals & Layout
- **Global Overview**: The homepage will feature a modest "Site Statistics" section summarizing all the collections combined (e.g., total items across all 10 collections showcased).
- **Collection Pages**:
  - A new "At a Glance" statistics bar at the top (e.g., "Items in collection: 250 | Date Range: 1914-1918").
  - A "Data Explorer" section below the description.
  - Each collection will receive *one* highly tailored visualization:
    - Example: *Allied Posters of World War I* gets a bar chart/timeline showing the distribution of posters by year (1914-1918).
    - Example: *City Parks Association Photographs* (or another relevant collection) gets a map plotting the locations of the photographs around Philadelphia.

## 4. Implementation Plan
1. **Data Enrichment**: Update `data/collections.json` with aggregate stats and add sample visualization data for 1-2 pilot collections (e.g., WWI Posters timeline, City Parks map).
2. **Setup Dependencies**: Add Chart.js and/or Leaflet to the site's asset pipeline or via CDN.
3. **Template Updates**: Modify the `layouts/collections/single.html` template to render the new statistics UI and inject the visualization data into JavaScript variables.
4. **Visualization Logic**: Write the JavaScript to initialize and render the charts/maps based on the injected data.
5. **Review & Polish**: Ensure it looks premium and responsive.
