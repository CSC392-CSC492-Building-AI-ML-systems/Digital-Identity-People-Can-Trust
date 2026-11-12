![cursor icon](https://luis.leiva.name/kme/cursors.png)

# The Attentive Cursor Dataset

The dataset includes mouse tracking recordings from a crowdsourcing study that measured user attention to web advertisements.


## Mouse tracking log files

The `logs/` folder has all the log files, as recorded by the [evtrack](https://github.com/luileito/evtrack) software.

  Here you can find space-delimited files (CSV) with information about each event type
  and XML files with information about the user’s browser metadata (e.g. viewport size, user agent string).

  Each CSV file has 8 columns:
  * `cursor`: (int) This column is always `0` because all participants used a regular computer mouse.
  * `timestamp`: (int) Timestamp of the event, with millisecond precision.
  * `xpos`: (float) X position of the mouse cursor.
  * `ypos`: (float) Y position of the mouse cursor.
  * `event`: (string) Browser's event name; e.g. `load`, `mousemove`, `click`, etc.
  * `xpath` (string) Target element that relates to the event, [in XPath notation](https://en.wikipedia.org/wiki/XPath).
  * `attrs` (string) Optional. Element attributes, if any.
  * `extras`: (string) Optional. A JSON string with Euclidean distances to different reference points of the ad's bounding box.

  ```csv
  cursor timestamp xpos ypos event xpath attrs extras
  0 1405503114382 0 0 load / {}
  ...
  ```

  **Note:** For events that do *not* relate to any mouse event (e.g. `load` or `blur`), the `xpos` and `ypos` column values are `0`.

  Example of the XML files:
  ```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <data>
   <date>Tue, 02 Oct 2018 03:31:26 +0200</date>
   <ua>Mozilla/5.0 (Windows NT 10.0; WOW64; rv:62.0) Gecko/20100101 Firefox/62.0</ua>
   <screen>1366x768</screen>
   <window>1366x632</window>
   <document>1349x2064</document>
   <task>5npsk114ba8hfbj4jr3lt8jhf5-dd-top_left</task>
  </data>
  ```


## Ground-truth labels

The (tab-delimited) `groundtruth.tsv` file has 4 columns:

  - `user_id`: (string) Participant's ID.
  - `ad_clicked`: (int) Whether the participant clicked on the ad (1) or not (0).
  - `attention`: (int) Self-reported attention score, in 1-5 Likert-type scale (1 denotes no attention).
  - `log_id`: (string) Mouse tracking log ID.

  Example:
  ```tsv
  user_id	ad_clicked	attention	log_id
  5npsk114ba8hfbj4jr3lt8jhf5	0	4	20181002033126
  ...
  ```


## User’s demographics and stimuli

The (tab-delimited) `participants.tsv` file has 12 columns:

  - `user_id`: (string) Participant's ID.
  - `country`: (string) Participant's country, in [ISO-3 format](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3).
  - `education`: (int) Level of education, according to 6 bins (see table below).
  - `age`: (int) Participant's age, according to 9 bins (see table below).
  - `income`: (int) Level of income, according to 8 bins (see table below).
  - `gender`: (string) Participant's gender.
  - `ad_position`: (string) Ad stimulus position.
  - `ad_type`: (string) Ad stimulus type.
  - `ad_category`: (string) Ad stimulus category.
  - `serp_id`: (string) Ad SERP identifier.
  - `query`: (string) Ad stimulus query.
  - `log_id`: (string) Mouse tracking log ID.

  | Bin | Education   | Age   | Income   |
  |-----|-------------|-------|----------|
  | 1   | High school | 18-23 | 25K      |
  | 2   | College     | 24-29 | 25-34K   |
  | 3   | Bachelor's  | 30-35 | 35-49K   |
  | 4   | Graduate    | 36-41 | 50-74K   |
  | 5   | Master's    | 42-47 | 75-99K   |
  | 6   | Doctorate   | 48-53 | 100-149K |
  | 7   |             | 54-59 | 150-249K |
  | 8   |             | 60-65 | +250K    |
  | 9   |             |   +66 |          |

  Example:
  ```tsv
  user_id	country	education	age	income	gender	ad_position	ad_type	ad_category	serp_id	query	log_id
  5npsk114ba8hfbj4jr3lt8jhf5	PHL	3	3	1	male	top-left	dd	Computers & Electronics	tablets	tablets	20181002033126
  ...
  ```

  **Note:** Some users did not to provide some demographic info, in which case it is denoted by the `NA` value.

## Stimulus pages

The `serps/` folder contains page snapshots (in HTML format) of the search engine results.

  Each filename is the `serp_id` you can find in `participants.tsv`.


## Citation

If you use the dataset, please cite us using this bibliographic reference:

* Luis A. Leiva, Ioannis Arapakis. (2020) **The Attentive Cursor Dataset**. Front. Hum. Neurosci. 14.

```bibtex
@article{Leiva20:TACD,
  author  = {Luis A. Leiva and Ioannis Arapakis},
  title   = {The Attentive Cursor Dataset},
  year    = {2020},
  volume  = {14},
  journal = {Front. Hum. Neurosci.},
  doi     = {10.3389/fnhum.2020.565664},
  url     = {https://gitlab.com/iarapakis/the-attentive-cursor-dataset},
}
```

### Related papers

The attentive cursor dataset has been used in the papers listed below.
If you use the dataset in your own publications, 
please send us an email and we will include your paper here.

* Ioannis Arapakis, Antonio Penta, Hideo Joho, Luis A. Leiva. (2020).
  **A Price-per-attention Auction Scheme Using Mouse Cursor Information.**
  ACM Trans. Inf. Syst. 38(2).

* Ioannis Arapakis, Luis A. Leiva. (2020).
  **Learning Efficient Representations of Mouse Movements to Predict User Attention.**
  Proc. Intl. ACM SIGIR Conf. on Research and Development in Information Retrieval (SIGIR).

* Luis A. Leiva, Ioannis Arapakis, Costas Iordanou. (2021).
  **My Mouse, My Rules: Privacy Issues of Behavioral User Profiling via Mouse Tracking.**
  Proc. ACM SIGIR Conf. on Human Information Interaction and Retrieval (CHIIR).
  
* Lukas Brückner, Ioannis Arapakis, Luis A. Leiva. (2021).
  **When Choice Happens: A Systematic Examination of Mouse Movement Length for Decision Making in Web Search.**
  Proc. Intl. ACM SIGIR Conf. on Research and Development in Information Retrieval (SIGIR).
