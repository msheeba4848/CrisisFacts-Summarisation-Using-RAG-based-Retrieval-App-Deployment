# Discussion and Documentation

---

## 1. Objective
The project aimed to build a hybrid text retrieval and summarization system to process and query event-based datasets effectively. By combining **BM25** for keyword-based matching and **dense embeddings** for semantic recall, the system ensures both precision and contextual relevance. It provides two modes of interaction:
1. **Event and Class Label Summarization**: Allows users to select specific events and labels for targeted summaries.
2. **Custom Query Summarization**: Accepts user-defined queries to search across the dataset and generate summaries.

---

## 2. Hybrid Retrieval

### **Purpose**
Hybrid retrieval enhances the system by leveraging both:
- **BM25** for precise keyword matching.
- **Dense Embeddings** for capturing semantic similarity.

### **Implementation**
- **BM25**: Uses tokenized `cleaned_text` to compute keyword similarity.
- **Dense Embeddings**: Precomputed using `DistilBERT` to capture semantic relationships between queries and dataset entries.
- **Score Combination**:
  - Scores are normalized and combined using a weighted formula:
    \[
    \text{Combined Score} = \alpha \times \text{BM25 Score} + (1 - \alpha) \times \text{Dense Score}
    \]

---

## 3. Query Normalization

### **Purpose**
Standardizing user queries ensures better mapping to dataset events and labels by handling variations in format, filler phrases, and punctuation.

### **Enhancements**
- Removal of filler words (e.g., "Can you find information on...") and unnecessary punctuation (e.g., "???").
- Standardization to lowercase and snake_case format for consistent matching.

### **Implementation**
#### Function: `normalize_query`
- **Input**: Raw user query.
- **Process**:
  - Lowercase conversion.
  - Removal of filler phrases and special characters.
  - Conversion to snake_case for compatibility with dataset keys.
- **Output**: Normalized query.

---

## 4. Human-Readable Interaction

### **Purpose**
Enhancing user experience by converting technical terms into human-readable formats:
- **Event Names**: Transforming `2015_nepal_earthquake` into `2015 Nepal Earthquake`.
- **Class Labels**: Converting `donation_and_volunteering` into `Donation and Volunteering`.

### **Impact**
- Intuitive interaction for users unfamiliar with the dataset's technical details.
- Dynamic conversion ensures compatibility with dataset fields.

### **Implementation**
#### Functions:
1. **`humanize_events`**:
   - Converts snake_case events to readable formats.
   - Adds contextual replacements for event types (e.g., `_earthquake` → ` Earthquake`).
2. **`humanize_labels`**:
   - Converts snake_case class labels into title-cased human-readable text.

---

## 5. Summarization

### **Purpose**
Generate concise and meaningful summaries from the retrieved dataset rows.

### **Model**
- **Model Used**: `facebook/bart-large-cnn`.
- **Advantages**:
  - Extractive summarization ensures relevant parts of the text are preserved.
  - Handles large input sequences effectively with truncation.

### **Implementation**
#### Function: `summarize_texts`
- **Input**: DataFrame of filtered rows, `max_length`, and `min_length`.
- **Process**:
  1. Concatenate relevant `cleaned_text` entries.
  2. Truncate the text to fit model constraints.
  3. Pass the text to the summarizer for generating outputs.
- **Output**: Summarized text.

---

## 6. Modes of Interaction

### **Mode 1: Event and Class Label Summarization**
Users specify an event and a class label to retrieve and summarize relevant data.

#### Workflow:
1. **Event Selection**:
   - Retrieve top matching events using **hybrid retrieval**.
   - Display events in human-readable format.
2. **Class Label Selection**:
   - Display available labels for the selected event.
   - Dynamically filter rows by event and label.
3. **Generate Summary**:
   - Summarize `cleaned_text` of filtered rows.

---

### **Mode 2: Custom Query Summarization**
Users input a free-text query, and the system searches across all events and labels to generate a summary.

#### Workflow:
1. **Query Normalization**:
   - Standardize the input query.
2. **Hybrid Retrieval**:
   - Retrieve top rows based on combined BM25 and dense scores.
3. **Dynamic Filtering**:
   - Filter retrieved rows for matching events or labels.
4. **Summarization**:
   - Summarize concatenated `cleaned_text` from the top results.

---
