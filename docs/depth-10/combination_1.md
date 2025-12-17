# Combination 1 (HS → CD → DS → MP → HS → MP → MP → MP → MP → MP)

## **IN-CONTEXT EXEMPLAR 1**

**Context:**
Jordan was told that if he submits his draft early, then his advisor gives feedback. His advisor giving feedback always leads to his paper improving. If his mentor attends the research meeting, then she offers guidance. Either Jordan submits early or his mentor attends the meeting. Whenever guidance is offered, Jordan refines his outline. If Jordan refines his outline, then he becomes more confident, and if he becomes more confident, then he is more prepared. Whenever Jordan is more prepared, he performs well. When Jordan performs well, he gains recognition. If he gains recognition, he receives new opportunities. Whenever he receives new opportunities, he achieves his career goals.

{P}: Jordan submits his draft early.
{Q}: Jordan's advisor gives feedback.
{R}: Jordan's paper improves.
{S}: Jordan's mentor attends the research meeting.
{T}: Jordan's mentor offers guidance.
{U}: Jordan refines his outline.
{V}: Jordan becomes more confident.
{W}: Jordan becomes more prepared.
{X}: Jordan performs well.
{Y}: Jordan gains recognition.
{Z}: Jordan receives new opportunities.
{Ω}: Jordan achieves his career goals.

**Question:** If R is not true, then is Ω true?

---

## **IN-CONTEXT EXEMPLAR 2**

**Context:**
If Maya practices violin regularly, then her tone improves, and whenever her tone improves, the ensemble blends better. If the conductor gives special coaching, then the section becomes more cohesive. Either Maya practices regularly or the conductor gives special coaching. When the section is cohesive, Maya gains clarity in her bowing. If she gains clarity, then she gains expressiveness, and if she gains expressiveness, then her performance strengthens. Whenever her performance strengthens, the concert result is positive. When the concert result is positive, the audience responds enthusiastically. If the audience responds enthusiastically, the orchestra gets invited for more performances. Whenever they get invited for more performances, the ensemble builds lasting reputation.

{P}: Maya practices violin regularly.
{Q}: Maya's tone improves.
{R}: The ensemble blends better.
{S}: The conductor gives special coaching.
{T}: The section becomes more cohesive.
{U}: Maya gains clarity in her bowing.
{V}: Maya gains expressiveness.
{W}: Maya's performance strengthens.
{X}: The concert result is positive.
{Y}: The audience responds enthusiastically.
{Z}: The orchestra gets invited for more performances.
{Ω}: The ensemble builds lasting reputation.

**Question:** If R is not true, then is Ω true?

---

## **IN-CONTEXT EXEMPLAR 3**

**Context:**
If Ravi studies early in the morning, then he grasps concepts faster. Whenever he grasps concepts faster, he solves harder exercises. If his classmate joins study group sessions, then she asks helpful questions. Either Ravi studies early or his classmate joins the sessions. Whenever helpful questions are asked, Ravi revises the material. If he revises the material, then he organizes his notes, and if he organizes his notes, then he remembers key ideas better. Whenever he remembers key ideas better, he performs strongly on quizzes. When he performs strongly on quizzes, his semester grade improves. If his semester grade improves, he qualifies for the honors program. Whenever he qualifies for the honors program, he secures better internship offers.

{P}: Ravi studies early in the morning.
{Q}: Ravi grasps concepts faster.
{R}: Ravi solves harder exercises.
{S}: Ravi's classmate joins the study group sessions.
{T}: Ravi's classmate asks helpful questions.
{U}: Ravi revises the material.
{V}: Ravi organizes his notes.
{W}: Ravi remembers key ideas better.
{X}: Ravi performs strongly on quizzes.
{Y}: Ravi's semester grade improves.
{Z}: Ravi qualifies for the honors program.
{Ω}: Ravi secures better internship offers.

**Question:** If R is not true, then is Ω true?

---

## **IN-CONTEXT EXEMPLAR 4**

**Context:**
If Lila submits her grant draft, then her supervisor reviews it. Whenever the supervisor reviews it, the proposal becomes stronger. If the finance officer approves the preliminary budget, then the team can move forward. Either Lila submits her draft or the officer approves the budget. When the team moves forward, Lila creates a detailed timeline. If she creates a detailed timeline, then she refines her milestones, and if she refines her milestones, then she improves her planning. Whenever her planning improves, her project proposal is accepted. When her proposal is accepted, she receives initial funding. If she receives initial funding, she can hire research assistants. Whenever she hires research assistants, her lab productivity doubles.

{P}: Lila submits her grant draft.
{Q}: Lila's supervisor reviews the draft.
{R}: The proposal becomes stronger.
{S}: The finance officer approves the preliminary budget.
{T}: The team can move forward.
{U}: Lila creates a detailed timeline.
{V}: Lila refines her milestones.
{W}: Lila improves her planning.
{X}: Lila's project proposal is accepted.
{Y}: Lila receives initial funding.
{Z}: Lila can hire research assistants.
{Ω}: Lila's lab productivity doubles.

**Question:** If R is not true, then is Ω true?

---

## **IN-CONTEXT EXEMPLAR 5**

**Context:**
If Ben calibrates the telescope, then the images become clearer. When the images become clearer, the analysis quality increases. If the weather station predicts stable skies, then the observing session proceeds smoothly. Either Ben calibrates the telescope or the weather station predicts stable skies. When the session proceeds smoothly, Ben logs high-quality data. If he logs high-quality data, then he identifies key patterns, and if he identifies key patterns, then he interprets results accurately. Whenever he interprets results accurately, the scientific report turns out well. When the report turns out well, peer reviewers give positive feedback. If reviewers give positive feedback, the paper gets published. Whenever the paper gets published, Ben's research reputation grows.

{P}: Ben calibrates the telescope.
{Q}: The images become clearer.
{R}: The analysis quality increases.
{S}: The weather station predicts stable skies.
{T}: The observing session proceeds smoothly.
{U}: Ben logs high-quality data.
{V}: Ben identifies key patterns.
{W}: Ben interprets results accurately.
{X}: The scientific report turns out well.
{Y}: Peer reviewers give positive feedback.
{Z}: The paper gets published.
{Ω}: Ben's research reputation grows.

**Question:** If R is not true, then is Ω true?

---

## Reasoning Trace for Combination 1

**Step-by-step reasoning (using Exemplar 1):**

1. **Given:** ¬R (Jordan's paper does not improve)
2. **Step 1 (HS):** From (P→Q) ∧ (Q→R), derive (P→R)
3. **Step 2 (MT on derived):** From (P→R) ∧ ¬R, derive ¬P (Jordan did not submit early)
4. **Step 3 (CD setup):** We have (P→R) ∧ (S→T) ∧ (P∨S)
5. **Step 4 (DS):** From (P∨S) ∧ ¬P, derive S (mentor attends)
6. **Step 5 (MP):** From (S→T) ∧ S, derive T (mentor offers guidance)
7. **Step 6 (MP):** From (T→U) ∧ T, derive U (refines outline)
8. **Step 7 (HS):** From (U→V) ∧ (V→W), derive (U→W)
9. **Step 8 (MP):** From (U→W) ∧ U, derive W (more prepared)
10. **Step 9 (MP):** From (W→X) ∧ W, derive X (performs well)
11. **Step 10 (MP):** From (X→Y) ∧ X, derive Y (gains recognition)
12. **Step 11 (MP):** From (Y→Z) ∧ Y, derive Z (receives opportunities)
13. **Step 12 (MP):** From (Z→Ω) ∧ Z, derive Ω (achieves career goals)

**Answer: Yes (Ω is true)**
