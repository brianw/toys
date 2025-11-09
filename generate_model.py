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
