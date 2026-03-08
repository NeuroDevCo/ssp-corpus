#!/bin/bash

OUTPUT_FILE="language_speakers.csv"
USER_AGENT="BashScript/1.0 (https://example.org)"
TEMP_JSON="response.json"

# The SPARQL Query
read -r -d '' SPARQL_QUERY << 'EOF'
SELECT DISTINCT ?item ?numberOfSpeakers ?speakersTime ?iso639_1 ?iso639_3 ?itemLabel ?itemLabel_en 
WHERE {
  ?item wdt:P31 wd:Q1288568;      # Any instance of a modern (current use) language
      wdt:P220 ?iso639_3;         # that has an ISO 639-3 code

  # Require number of speakers, but as a statement so we can get the "timestamp" of the property:
  ?item p:P1098 ?numberOfSpeakersStatement .
  ?numberOfSpeakersStatement ps:P1098 ?numberOfSpeakers.
  optional {
    ?numberOfSpeakersStatement pq:P585 ?speakersTime.
    FILTER(datatype(?speakersTime) = xsd:dateTime). # exclude unknown values
  }

  # Get the native language label, if there is one, trying ISO 639-1 and -3:
  # (Can we match language of labels using wikidata identifiers?)
  optional {
    ?item wdt:P218 ?iso639_1.
    ?item rdfs:label ?itemLabel_1 . FILTER(lang(?itemLabel_1)=?iso639_1)
  }
  optional {
    ?item rdfs:label ?itemLabel_3 . FILTER(lang(?itemLabel_3)=?iso639_3)
  }
  bind(coalesce(?itemLabel_3, ?itemLabel_1) as ?itemLabel)

  # Get the label in English, if there is one:
  optional {
    ?item rdfs:label ?itemLabel_en . FILTER(lang(?itemLabel_en)='en')
  }

}
ORDER BY DESC(?numberOfSpeakers) DESC(?speakersTime)
EOF

echo "Downloading data..."

# 1. Download JSON to a temporary file
# -s = Silent, -S = Show errors, -L = Follow redirects
# 2>/dev/null ensures no progress text leaks into the file
curl -s -S -L -H "User-Agent: $USER_AGENT" \
  -G "https://query.wikidata.org/sparql" \
  --data-urlencode "query=$SPARQL_QUERY" \
  --data-urlencode "format=json" \
  -o "$TEMP_JSON" 2>/dev/null

# 2. Check if we actually got JSON
if head -n 1 "$TEMP_JSON" | grep -q "json"; then
    echo "Data received. Converting to CSV..."
    
    # 3. Parse JSON to CSV
    jq -r '
      ["item", "numberOfSpeakers", "speakersTime", "iso639_1", "iso639_3",  "itemLabel", "itemLabel_en"],
      (.results.bindings[] | [
        .item.value, 
        .numberOfSpeakers.value, 
        .speakersTime.value // "", 
        .iso639_1.value // "", 
        .iso639_3.value, 
        .itemLabel.value, 
        .itemLabel_en.value // ""
      ] | @csv)
    ' "$TEMP_JSON" > "$OUTPUT_FILE"
    
    echo "Done. Results saved to $OUTPUT_FILE"
    rm "$TEMP_JSON"
else
    echo "Error: The server did not return valid JSON."
    echo "Check the contents of $TEMP_JSON for details."
    head -c 500 "$TEMP_JSON"
fi
