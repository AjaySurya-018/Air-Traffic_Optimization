# ✈️ Air-Traffic Optimization

> **Predictive & Prescriptive Analytics for Flight Delays and Ground Operations**

A data-driven system that combines time-series forecasting and optimization algorithms to reduce flight delays, minimize taxi times, and optimize route selection for fuel efficiency.

---

## 🎯 Overview

Airline delays cost billions annually and frustrate passengers. This project tackles the problem from two angles:

**🔮 Predictive Analytics**  
Forecast route-level arrival delays using Prophet time-series models to enable proactive scheduling decisions.

**⚙️ Prescriptive Optimization**  
Optimize gate assignments, taxi operations, and schedules using linear/integer programming (PuLP) and nonlinear optimization (scipy).

### Key Results
- **Gate Assignment**: Reduce total taxi time by optimizing flight-to-gate assignments
- **Schedule Buffering**: Minimize arrival delays through intelligent schedule adjustments  
- **Route Selection**: Optimize flight selection under distance and capacity constraints

---

## 📊 Dataset

**Source**: Historical flight records (2003–2008)  
**Files**: 
- `data/Final_data.csv` — Cleaned flight records
- `flights.db` — SQLite database with indexed queries

**Key Fields**:
```
Temporal: Year, Month, DayOfWeek, DayofMonth
Flight: UniqueCarrier, FlightNum, Origin, Dest, Distance
Times: CRSElapsedTime, ActualElapsedTime, TaxiIn, TaxiOut
Delays: DepDelay, ArrDelay, CarrierDelay, WeatherDelay, NASDelay
```

---

## 🔬 Methodology

### 1️⃣ Forecasting with Prophet

Train additive seasonal models on route-aggregated annual delays:

```python
# Aggregate by route and year
route_data['ds'] = pd.to_datetime(route_data['Year'], format='%Y')
route_data = route_data.rename(columns={'AvgArrDelay': 'y'})

# Fit and forecast
model = Prophet(yearly_seasonality=True, interval_width=0.95)
model.fit(route_data[['ds', 'y']])
future = pd.DataFrame({'ds': pd.date_range('2008', '2015', freq='Y')})
forecast = model.predict(future)
```

**Outputs**: Mean predictions (`yhat`) with 95% confidence intervals

### 2️⃣ Optimization Models

#### Gate Assignment (Integer Programming)
Minimize total taxi time by optimally assigning flights to gates:

```
minimize: Σ x_fg × (TaxiIn_f + TaxiOut_f)
subject to:
  • Each flight assigned to exactly one gate
  • Each gate handles at most one flight
  • x_fg ∈ {0,1}
```

#### Route Selection (Linear Programming)
Select flights to operate under distance/capacity constraints while minimizing fuel consumption.

#### Schedule Adjustment (Nonlinear Optimization)
Adjust scheduled elapsed times within buffers to minimize arrival delays:

```python
def objective(adjustments):
    adjusted_schedule = base_schedule + adjustments
    delays = np.maximum(0, actual_times - adjusted_schedule)
    return delays.sum()

# Bounded optimization
scipy.optimize.minimize(objective, x0, bounds=[(min_adj, max_adj)])
```

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[📁 Raw Data<br/>Final_data.csv] --> B[🗄️ SQLite DB<br/>flights.db]
    B --> C[🔧 Data Prep<br/>Aggregation & Filtering]
    C --> D[📈 Prophet<br/>Forecasting]
    C --> E[⚡ Optimization<br/>PuLP + scipy]
    D --> F[🎯 Predictions<br/>yhat + intervals]
    F --> E
    E --> G[📋 Solutions<br/>Gate plans, schedules]
    G --> H[📊 Streamlit Apps<br/>Interactive Dashboard]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style D fill:#f3e5f5
    style E fill:#e8f5e9
    style H fill:#fce4ec
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/AjaySurya-018/Air-Traffic_Optimization.git
cd Air-Traffic_Optimization

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Applications

```bash
# Main dashboard
streamlit run App.py

# Specific modules
streamlit run Taxi_time_optimization.py
streamlit run Flight_average_delay_forecasting.py
streamlit run Route_Fuel_Optimization.py
```

---

## 📈 Performance Metrics

| Metric | Evaluation Method |
|--------|-------------------|
| **Forecasting** | MAE, RMSE, prediction interval coverage |
| **Optimization** | Total taxi time, arrival delay reduction, fuel savings |
| **Operational** | Feasibility rate, constraint violations |

### Example Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Total Taxi Time (min) | 4,320 | 3,120 | **↓ 28%** |
| Total Arrival Delay (min) | 9,800 | 7,500 | **↓ 23%** |

---

## 💡 Key Insights

✅ **Route-level forecasting** enables proactive schedule adjustments for seasonal patterns  
✅ **Gate optimization** significantly reduces taxi times and associated fuel burn  
✅ **Schedule buffering** aligns planned times with realistic operations  
✅ **Delay decomposition** (Carrier, Weather, NAS) reveals both controllable and exogenous factors

---

## ⚠️ Limitations & Future Work

**Current Limitations**:
- Annual aggregation loses granular patterns
- Static optimization doesn't model temporal gate conflicts
- Simplified routing model lacks fleet/crew constraints

**Roadmap**:
- [ ] Fine-grained forecasting (daily/weekly with weather covariates)
- [ ] Time-window interval scheduling for dynamic gate assignment
- [ ] Causal inference for delay attribution
- [ ] Real-time integration with ATC systems

---

## 🛠️ Tech Stack

**Core**: Python 3.8+  
**Forecasting**: Prophet, pandas  
**Optimization**: PuLP (CBC solver), scipy.optimize  
**Visualization**: Streamlit, matplotlib, seaborn  
**Database**: SQLite

---

## 📚 References

- Taylor, S. J., & Letham, B. (2018). *Forecasting at scale*. Prophet.
- Belobaba, P., et al. (2009). *The Global Airline Industry*.
- Operations research literature on airport ground operations and gate assignment MIP formulations.
