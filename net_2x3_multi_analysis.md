
# 2x3 Environment Logic Analysis

This document analyzes the logic within `envs/net_2x3.py` to explain how the simulation maps to real-world supply chain concepts.

## AREA 1: The Echelon Mapping (Who is Who?)

**Topic:** Echelon Hierarchy
**Code Snippet:**
```python
297: cur_demmand = [[[self.demand_list[0][self.step_num], 0], [0, self.demand_list[1][self.step_num]]]]
...
299: for i in range(1, self.level_num):
300:     de = []
301:     de.append([action_[(i-1)*2][0] + ...
```
**Plain English Explanation:**
The simulation maps **Level 0** to the **Retailers** (closest to the customer) and **Level 2** to the **Factories** (furthest away). 
We know this because the code injects the customer demand (`self.demand_list`) directly into index 0 (`cur_demmand`). The demand for higher levels (1 and 2) is then calculated based on the orders (`action_`) placed by the level below them (`i-1`). Thus, goods flow down (2 -> 1 -> 0) and orders flow up (0 -> 1 -> 2).

- **Index 0**: Retailers (Face the customer)
- **Index 1**: Wholesalers/Distributors
- **Index 2**: Manufacturers/Factories

## AREA 2: Multi-Product Logic (SKU Check)

**Topic:** Inventory Structure
**Code Snippet:**
```python
112: self.inventory = [[S_I, S_I] for i in range(LEVEL_NUM)]
```
**Plain English Explanation:**
The system is **Single-Product** but **Multi-Agent**. 
`self.inventory` is a list of lists. `LEVEL_NUM` is 3, so there are 3 "rows." Each row has `[S_I, S_I]`, meaning there are **2 distinct agents** at each level.
This represents a network topology where there are 2 parallel supply chains (Agent A and Agent B) at each echelon level, but they are all managing the same type of abstract "unit" or product. It is *not* a multi-SKU system where one agent manages separate stocks for "Apples" and "Oranges."

## AREA 3: Cost Logic (Holding vs. Ordering)

**Topic:** Cost Calculation
**Code Snippet:**
```python
330: b_c = B[i*2+j]*np.sum(self.backlog[(i)*2+j])
331: reward = - actual_order[0]*C[i][0] - actual_order[1]*C[i][1] - self.inventory[i][j]*H[i] - b_c
```
**Plain English Explanation:**
The simulation calculates the "Reward" as the negative sum of three costs. The goal is to get this number as close to zero as possible.
1.  **Ordering Cost**: `actual_order[0]*C[i][0]...`. You pay for every item you order. Uniquely, ordering from Supplier 1 (Cost 0.1) is cheaper than Supplier 2 (Cost 0.2), creating a trade-off.
2.  **Holding Cost**: `self.inventory[i][j]*H[i]`. You pay $1.00 (`H[i]`) for every unit sitting in your warehouse at the end of the day.
3.  **Backlog Cost**: `b_c`. You pay $1.00 (`B`) for every unit of customer demand you failed to fulfill.

## AREA 4: Observation Space (What they see)

**Topic:** Agent Vision
**Code Snippet:**
```python
246: arr = np.array([self.inventory[i][j], 
                     self.backlog[i*2+j][0] + self.backlog[i*2+j][1], 
                     self.backlog[(i+1)*2][j]+self.backlog[(i+1)*2+1][j], 
                     action[(i-1)*2][j]+action[(i-1)*2+1][j]] + 
                     self.order[i][j])
```
**Plain English Explanation:**
Each agent sees a specific "Dashboard" of 8 numbers (`OBS_DIM = 8`).
1.  **My Inventory**: How much stock I have right now.
2.  **My Backlog**: How many orders I currently owe to my customers.
3.  **Supplier's Backlog**: How much my supplier owes me (indicates if the supplier is in trouble).
4.  **Customer Demand**: The total orders my customers placed yesterday.
5.  **In-Transit 1**: Shipments arriving in 1 day.
6.  **In-Transit 2**: Shipments arriving in 2 days.
7.  **In-Transit 3**: Shipments arriving in 3 days.
8.  **In-Transit 4**: Shipments arriving in 4 days.

## AREA 5: The "Bullwhip Killer" (Global Reward)

**Topic:** Cooperation Mechanism
**Code Snippet:**
```python
25: ALPHA = 0.75
...
276: processed_rewards = [[ALPHA*i+(1-ALPHA)*np.mean(reward)] for i in reward]
```
**Plain English Explanation:**
This is the **Critical Mechanism** for reducing the Bullwhip Effect.
Normally, an agent only cares about its own costs (`i`). However, this line forces them to share the pain.
The final score (`processed_rewards`) is a weighted mix:
-   **75% (`ALPHA`)** comes from your **Own Personal Cost**.
-   **25% (`1-ALPHA`)** comes from the **Average Cost of the Entire Team**.

**How this kills the Bullwhip Effect:**
If the Retailer panic-orders too much, it might not hurt the Retailer immediately (they have stock), but it will crush the Factory with huge holding or overtime costs. In a selfish system, the Retailer wouldn't care. 
Here, if the Factory suffers a massive cost, the "Team Average" drops significantly. This pulls down the Retailer's score too (`0.25 * BigPenalty`). Therefore, the Retailer learns *not* to panic-order because hurting the Factory eventually hurts the Retailer.
