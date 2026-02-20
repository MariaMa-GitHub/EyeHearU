# A2 — Datasets

## Part One: Aspirational Datasets

If we had a magic wand, the following list of datasets and data schemas is what our team would need to be successful in this project:

---

### 1. Isolated Word-Level ASL Sign Videos Training Dataset

We need many short video clips of individual ASL signs performed by diverse signers such that our video classifier can learn the spatial and temporal patterns that distinguish one sign from another.

**Ideally:**
- The glosses not only cover the target vocabulary of our project but also have a large scale of more than 2000 words.
- There are more than 100 videos per gloss in order to capture signer variation.
- There are more than 50 unique signers having balanced ages, genders, hand sizes, etc.
- Each video clip is tightly trimmed to the sign only, with no useless idle time.
- Each video clip has high resolution and reasonable frame rate, yet is not bulky in file size.
- The video clips contain a mix of clean and real-world backgrounds (home, office, school, restaurant, clinic, etc.)
- The video clips use a variety of lighting, camera angles, and camera types.

**Ideal schema:**

| Field | Type | Description | Example |
|---|---|---|---|
| video_id | string | Unique identifier for video clip | "wlasl_hello_001" |
| video_path | string | Path to video clip file | "videos/hello/wlasl_hello_001.mp4" |
| gloss | string | English label for the ASL sign | "hello" |
| category | string | Scenario category | "greeting" |
| signer_id | string | Unique identifier for the signer | "signer_009" |
| signer_demographics | object | Characteristics of the signer | {"age_range": "20-30", "skin_tone": "medium", "used_hand": "right", "gender": "female"} |
| duration | float | Video clip duration (in seconds) | 1.9 |
| fps | int | Video clip frame rate | 30 |
| num_frames | int | Total number of frames in video clip | 42 |
| resolution | string | Video clip resolution | "1280x720" |
| env_type | string | Type of environment | "clean studio" |
| light_type | string | Type of lighting | "daylight" |
| camera_type | string | Type of camera | "phone front" |

---

### 2. Mobile-Captured Evaluation Dataset

We need many short video clips of individual ASL signs recorded on mobile phone cameras in the way our app will actually be used (front-facing iPhone camera, held at arm's length, in real environments like a restaurant or school) such that we can test whether our model accuracy holds up under real deployment conditions.

**Ideally:**
- The glosses not only cover the target vocabulary of our project but also have a large scale of more than 2000 words.
- There are more than 15 videos per gloss in order to capture signer variation.
- There are more than 10 unique signers (which should not be overlapping with the ones in the training dataset) having balanced ages, genders, hand sizes, etc.
- Each video clip is captured using a front-facing mobile phone camera (selfie mode).
- Each video clip is tightly trimmed to the sign only, with no useless idle time.
- Each video clip has high resolution and reasonable frame rate, yet is not bulky in file size.
- The video clips contain various real-world backgrounds (home, office, school, restaurant, clinic, etc.)
- The video clips use a variety of lighting and camera angles.

**Ideal schema:**

| Field | Type | Description | Example |
|---|---|---|---|
| video_id | string | Unique identifier for video clip | "wlasl_hello_001" |
| video_path | string | Path to video clip file | "videos/hello/wlasl_hello_001.mp4" |
| gloss | string | English label for the ASL sign | "hello" |
| category | string | Scenario category | "greeting" |
| signer_id | string | Unique identifier for the signer | "signer_001" |
| signer_demographics | object | Characteristics of the signer | {"age_range": "20-30", "skin_tone": "light", "used_hand": "left", "gender": "male"} |
| duration | float | Video clip duration (in seconds) | 1.1 |
| fps | int | Video clip frame rate | 30 |
| num_frames | int | Total number of frames in video clip | 38 |
| resolution | string | Video clip resolution | "1280x720" |
| env_type | string | Type of environment | "restaurant" |
| light_type | string | Type of lighting | "daylight" |
| camera_type | string | Type of camera | "phone front" |

---

### 3. Scenario-Specific Phrase Sequence Evaluation Dataset

We need many video clips of multi-sign phrases used in our everyday life—including our project's target scenarios (e.g., "water please", "eat more")—such that we can evaluate whether our app can support realistic turn-by-turn interactions.

**Ideally:**
- The video clips contain full phrases along with per-sign timestamp annotations.
- There are more than 30 common phrase sequences.
- There are more than 1 sign per phrase.
- There are more than 10 unique signers (which should not be overlapping with the ones in the training dataset) having balanced ages, genders, hand sizes, etc.
- Each video clip is captured using a front-facing mobile phone camera (selfie mode).
- Each video clip is tightly trimmed to the signs only, with no useless idle time.
- Each video clip has high resolution and reasonable frame rate, yet is not bulky in file size.
- The video clips contain various real-world backgrounds (home, office, school, restaurant, clinic, etc.)
- The video clips use a variety of lighting and camera angles.

**Ideal schema:**

| Field | Type | Description | Example |
|---|---|---|---|
| video_id | string | Unique identifier for video clip | "wlasl_restaurant_001" |
| video_path | string | Path to video clip file | "phrases/wlasl_restaurant_001.mp4" |
| glosses | list[string] | Ordered list of ASL sign labels | ["water", "please"] |
| timestamps | list[object] | Ordered list of temporal boundaries (in seconds) per sign within the video | [{"gloss": "water", "start": 0.2, "end": 1.1}, {"gloss": "please", "start": 1.2, "end": 1.9}] |
| signer_id | string | Unique identifier for the signer | "signer_005" |
| signer_demographics | object | Characteristics of the signer | {"age_range": "10-20", "skin_tone": "dark", "used_hand": "left", "gender": "female"} |
| duration | float | Video clip duration (in seconds) | 2.0 |
| fps | int | Video clip frame rate | 30 |
| num_frames | int | Total number of frames in video clip | 60 |
| resolution | string | Video clip resolution | "1280x720" |
| env_type | string | Type of environment | "restaurant" |
| light_type | string | Type of lighting | "daylight" |
| camera_type | string | Type of camera | "phone front" |

---

### 4. "Negative" Non-ASL Dataset

We need many video clips of "negative" examples (things that are not ASL signs: idle hand movement, scratching, gesturing) such that our model does not misclassify every random hand movement as an ASL sign.

**Ideally:**
- The video clips contain different random hand gestures, non-ASL gestures, idle motion, no hands visible, and ASL signs beyond simple everyday vocabulary.
- There are more than 500 video clips.
- There are more than 20 unique signers having balanced ages, genders, hand sizes, etc.

**Ideal schema:**

| Field | Type | Description | Example |
|---|---|---|---|
| video_id | string | Unique identifier for video clip | "negative_001" |
| video_path | string | Path to video clip file | "negatives/negative_001.mp4" |
| clip_type | string | Category of "negative" example | "random gesture" |
| duration | float | Video clip duration (in seconds) | 2.0 |
| fps | int | Video clip frame rate | 30 |
| num_frames | int | Total number of frames in video clip | 60 |

---
---

## Part Two: Reality Check

The following table contains a list of datasets we could use for our project:

| Relevant Item | Description | Commentary |
|---|---|---|
| **ASL Citizen** | The first crowdsourced isolated sign language video dataset, containing about 83.4k video recordings of 2.7k isolated signs from American Sign Language (ASL) recorded by 52 Deaf or hard of hearing signers (with consent). Each instance consists of a video file in .mp4 format, an associated gloss (or English transliteration), and an associated anonymous user identifier (1 of 52 signers). Instances are labeled as either "train" (training set), "val" (validation set), or "test" (test set), containing 40,154, 10,304, and 32,941 videos respectively. The data splits are stratified by user such that each user is unseen in the other data splits. The distribution of signers across splits was chosen to balance female-male gender ratio. | This dataset can be used as our project's primary training set, validation set, and test set, because it already has recommended data splits. We believe this dataset is most relevant to our project since: It has a larger vocabulary size (e.g. 2,731 from ASL Citizen vs 2,000 from WLASL-2000). The data is crowdsourced so it contains examples of everyday signers in everyday environments: videos are self-recorded and come from webcams in uncontrolled settings — varied lighting, cluttered backgrounds, different camera qualities. This is closer to how our app will be used in practice. |
| **WLASL (Word-Level ASL)** | The largest scraped video dataset for Word-Level American Sign Language (ASL) recognition, which features 2,000 common different words in ASL. Each video is annotated with a single gloss label and temporal boundaries (start/end frame of the sign). Comes in subsets: WLASL-100, WLASL-300, WLASL-1000, WLASL-2000. | This dataset could be used as a supplementary training set. It covers our target words (for instance, the subset WLASL-100 (containing the 100 most common glosses) overlaps significantly with our scenario vocabulary), and the temporal boundary annotations are critical for our video classifier — they allow us to extract tightly trimmed clips rather than having to process full YouTube videos (much more complicated). |
| **MS-ASL (Microsoft ASL)** | The first real-life large-scale sign language data set comprising over 25,000 annotated videos with over 200 signers, covering a large class count of 1000 signs in challenging and unconstrained real-life recording conditions. It has clean labelling and provides temporal annotations. Comes in subsets: MS-ASL100, MS-ASL200, MS-ASL500, MS-ASL1000 (each includes their own train, validation, and test sets). | This dataset could be used as a supplementary training set. It covers our target words (for instance, the subset MS-ASL100 (containing the 100 most common glosses) overlaps significantly with our scenario vocabulary), the sign videos are captured in a range of lightings and backgrounds (realistic settings), and its 222 distinct signers make it the most signer-diverse dataset available (this is important since our model must generalize to new users it has never seen). |

Compared to our list of aspirational datasets, we believe that the above three publicly available datasets contain most of everything we need. There is in fact a large number of video clips that are sign-isolated, shot in realistic settings (various lightings, backgrounds, camera types, etc.), recorded by diverse signers, and annotated with temporal boundaries that we can use for our project.

Unfortunately, based on our research, there aren't any "scenario-specific phrase sequence" ASL datasets or any "negative" non-ASL datasets that are publicly available for us to use.
