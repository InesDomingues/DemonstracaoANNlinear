# Age + BMI → Hypertension (Yes/No)

Interactive Streamlit demonstration of a **single sigmoid neuron** trained by gradient descent for binary classification. Adapted from the uploaded height/weight/sex example.

- **Inputs typed by participants:** Age (years), Height (cm), and Weight (kg).
- **Automatic feature calculation:** BMI = Weight (kg) / (Height (cm) / 100)², rounded to one decimal place for display/model input.
- **Model input features:** Age (years) and computed BMI (kg/m²); Height and Weight are **not** fed directly into the neuron.
- **Target label:** Hypertension — **Yes (1)** or **No (0)**.
- **Dataset:** Generated fictional observations (Age, Height, Weight, and BMI derived from Height/Weight) with a deliberately invented logistic relationship and random noise. It is **not** a real dataset, is **not** calibrated to Portugal or to any clinical population, and must **not** be interpreted as an epidemiological estimate.
- **Plots:** Age versus BMI, colour-coded by simulated hypertension label, with a changing linear decision boundary.
- **Network visualisation:** Input coefficients for Age and BMI, bias, sigmoid output, loss and training accuracy are updated after training clicks.
- **Prediction panel:** Choose hypothetical Age, Height and Weight. The app displays the BMI calculation and passes Age and BMI into the neuron, showing its probability and Yes/No classification at a threshold of 0.5.
- **Optional fictional cases:** Enter Age, Height and Weight (plus a fictional Hypertension label); BMI is calculated automatically. Records are added **only in the current Streamlit session**. No dataset or medical information is saved to disk.
