# Implementation Plan: Collections as Data

## Task 1: Global Setup & At a Glance Data
**Goal**: Add required CSS/JS libraries to the base layout and enrich `collections.json` with aggregate statistics. Update the single collection template to display "At a Glance" statistics.

### Steps:
1. Edit `layouts/_default/baseof.html` or equivalent base template to inject Chart.js (`https://cdn.jsdelivr.net/npm/chart.js`) and Leaflet CSS/JS (`https://unpkg.com/leaflet/dist/leaflet.css` and `https://unpkg.com/leaflet/dist/leaflet.js`).
2. Add the following fields to every collection in `data/collections.json`:
   - `total_items`: (e.g., "Total items: 250")
   - `date_range`: (e.g., "1914-1918")
3. Modify `layouts/collections/single.html` to add an "At a Glance" statistics bar below the description, utilizing these new fields.

### Verification:
Build the Hugo site (`hugo server`) and verify the libraries load in the `<head>` or bottom of body, and the "At a Glance" section appears on individual collection pages.

## Task 2: Temporal Visualization - Allied Posters of World War I
**Goal**: Add a timeline chart to the WWI Posters collection using Chart.js.

### Steps:
1. In `data/collections.json`, under the `wwi-posters` and `sheet-music` collections add a new `viz_type` attribute equal to `"bar_chart"` and a `viz_data` object containing a labels array (years) and a data array (counts).
2. Edit `layouts/collections/single.html` to check if `viz_type == "bar_chart"`. If so, render a `<canvas id="collectionChart"></canvas>` and add a `<script>` block that reads `viz_data` and initializes a `Chart.js` bar chart.

### Verification:
Load the WWI Posters page. Verify a visually appealing bar chart renders below the text, displaying items by year.

## Task 3: Geographic Visualization - City Parks Association
**Goal**: Add a map to the City Parks collection using Leaflet.

### Steps:
1. In `data/collections.json`, under the `city-parks` collection, add `viz_type: "map"` and `viz_data` containing an array of locations (lat, lng, title) representing where photos were taken.
2. Edit `layouts/collections/single.html` to check if `viz_type == "map"`. If so, render a `<div id="collectionMap" style="height: 400px;"></div>` and add a `<script>` block initializing a Leaflet map with markers for each location.

### Verification:
Load the City Parks page. Verify a map renders displaying the corresponding markers in Philadelphia.
