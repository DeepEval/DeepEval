# Human Evaluation Study: How to Open the Questionnaire Webpages

To open the questionnaire webpages correctly, please serve the `human_evaluation_study` directory through a local HTTP server rather than directly opening an HTML file via `file://`.

## Steps

1. Open a terminal and navigate to the following directory:

   ```bash
   DeepEval/evaluation/semantic_checking/human_evaluation_study
   ```

2. Start a local HTTP server:

   ```bash
   python3 -m http.server 8000
   ```

3. Open a web browser and enter the URL of the target questionnaire HTML file in the address bar. For example:

   ```text
   http://127.0.0.1:8000/questionnaires/index_gpt_4o_mini_oneshot_part1.html
   ```

## Notes

- Please do not open a questionnaire HTML file by double-clicking it, because the page loads additional JavaScript resources from the parent directory.
- The server should be started from the `human_evaluation_study` directory, not from the `questionnaires` directory. Otherwise, dependent files such as `index_template.js` may not be found.
