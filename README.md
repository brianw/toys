# Irish Property Price Predictor

A beautiful, zero-build single-page web app for predicting Irish residential property sale prices using XGBoost machine learning, running entirely client-side in the browser.

## Features

- 🚀 **Zero build system** - no npm, webpack, or bundlers required
- 🎨 **Modern UI** - clean, responsive design with Tailwind CSS
- 🔒 **100% client-side** - all predictions happen in your browser, no data sent to servers
- ⚡ **Instant predictions** - model runs via pure JavaScript (converted with m2cgen)
- 📱 **Mobile-friendly** - works on all devices

## Setup

### Step 1: Install m2cgen in your Python environment

```bash
pip install m2cgen
```

### Step 2: Generate the model.js file

In your Python repository with the trained XGBoost model, run:

```python
import m2cgen as m2c
import pickle

with open('path/to/your/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

js_code = m2c.export_to_javascript(model)

wrapper = f'''export function predict(features) {{
    const [eircode_routing_key, bedrooms, bathrooms, size_m, building_type, ber_rating, sale_date] = features;

    {js_code}

    return score(features);
}}
'''

with open('model.js', 'w') as f:
    f.write(wrapper)

print('model.js generated successfully!')
```

Or use the included `generate_model.py` script:

```bash
python generate_model.py
```

### Step 3: Copy model.js to this directory

Copy the generated `model.js` file to the same directory as `index.html`.

### Step 4: Open in browser

Simply open `index.html` in any modern web browser. No server required!

For development with live reload, you can use Python's built-in server:

```bash
python -m http.server 8000
```

Then open http://localhost:8000

## Model Input Features

The model expects the following features in this order:

1. **eircode_routing_key** (string) - First 3 characters of the Eircode
2. **bedrooms** (integer) - Number of bedrooms
3. **bathrooms** (integer) - Number of bathrooms
4. **size_m** (float) - Size in square metres
5. **building_type** (string) - One of: 'apartment', 'terraced', 'semi_detached', 'detached'
6. **ber_rating** (string) - One of: 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'E1', 'E2', 'F', 'G'
7. **sale_date** (integer) - Encoded as months since 2021-01-01 (automatically calculated from the date picker)

## How It Works

1. User fills in property details in the web form
2. JavaScript collects the form data and encodes the sale date
3. The `predict()` function from `model.js` runs the XGBoost model logic (pure JavaScript, no dependencies)
4. Result is displayed as a formatted Euro price

## Deployment

Deploy anywhere that can serve static files:

- **GitHub Pages**: Push to a repo and enable GitHub Pages
- **Netlify**: Drag and drop the folder
- **Vercel**: Connect your git repo
- **Any web server**: Just copy the files

No build step, no environment variables, no server-side code required.

## Maintenance

This app has **zero dependencies** and requires **zero maintenance**. The only time you need to update anything is when you retrain your XGBoost model - just regenerate `model.js` and replace the file.

No security patches, no dependency updates, no breaking changes. It just works.

## File Structure

```
.
├── index.html          # Main app (HTML + embedded JavaScript)
├── model.js            # Generated XGBoost model as pure JavaScript
├── generate_model.py   # Helper script to convert XGBoost to JavaScript
└── README.md          # This file
```

## Browser Compatibility

Works in all modern browsers that support:
- ES6 modules
- Fetch API
- CSS Grid/Flexbox

Tested in Chrome, Firefox, Safari, and Edge.
