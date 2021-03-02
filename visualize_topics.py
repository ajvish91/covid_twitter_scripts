#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import json
import sys
import random
from flask import Flask, jsonify, request, render_template

# app.py
app = Flask(__name__)

@app.route('/test')
def test_page():
    # look inside `templates` and serve `index.html`
    return render_template('index.html', data=data)


# Read topics file .xlsx format with a sheet named "Topics"
topic_df = pd.read_excel("topics.xlsx", sheet_name="Topics")
topic_df = topic_df.iloc[:][:10]

# Convert dataframe to list of dictionaries
topic_list = topic_df.to_dict("records")

# Sample some colours
colors = ["#7B241C", "#633974", "#A93226", "#145A32", "#76D7C4", "#B7950B", "#6E2C00", "#7B7D7D", "#B3B6B7", "#D35400", "#154360", "#0B5345"]
sam_col = random.sample(colors, topic_df.shape[0])

# Create Topic JSON compatible with Highcharts
topic_json = {"series": []}
for topic in topic_list:
    words = {}
    words["name"] = "Topic " + str(topic["Topic"] + 1)
    val = 1.0/(topic["Topic"] + 1) * 15000.0
    words["data"] = [{"name": topic["Word " + str(wrd)], "value" : float(topic["Relevance " + str(wrd)]) * val} for wrd in range(0, 15)]
    topic_json["series"].append(words)

for i, item in enumerate(topic_json["series"]):
    topic_json["series"][i]["color"] = sam_col[i]

""" 
FORMAT of data fed into HighCharts:

{"series": [{"name": "Topic 1", "data": [{"name": "this", "value": 255.00000000000003}, {"name": "is", "value": 195.0}, {"name": "the", "value": 195.0}, {"name": "first", "value": 165.0}, {"name": "topic", "value": 165.0}], "color": "#633974"}, {"name": "Topic 2", "data": [{"name": "this", "value": 210.0}, {"name": "is", "value": 135.0}, {"name": "the", "value": 127.50000000000001}, {"name": "second", "value": 82.5}, {"name": "topic", "value": 60.0}], "color": "#B7950B"}, {"name": "Topic 3", "data": [{"name": "this", "value": 70.0}, {"name": "is", "value": 65.0}, {"name": "the", "value": 60.0}, {"name": "third", "value": 60.0}, {"name": "topic", "value": 55.0}], "color": "#145A32"}

"""

data = json.dumps(topic_json)

print(data)