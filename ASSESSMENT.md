# Text-to-SQL Ollama Model Assessment

## Assessment Question: Text-to-SQL Model Implementation and Deployment

**Duration:** 60 minutes  
**Total Points:** 100 points

---

## Scenario
You are a Machine Learning Engineer tasked with implementing a text-to-SQL system for a mid-sized company. The system needs to convert natural language queries into SQL statements for their customer database. The company has limited computational resources (8GB RAM, no dedicated GPU) and needs a cost-effective solution that can be deployed locally.

---

## Part A: System Design and Model Selection (25 points)

### Question 1 (10 points)
Based on the project documentation, compare and contrast the following model options for your resource-constrained environment:

| Model Option | Base Size | Training Time | Use Case |
|--------------|-----------|---------------|----------|
| CodeLlama-7B | 13GB | 2-3 hours | ? |
| CodeT5+ 770M | 770MB | ? | ? |
| CodeT5+ 220M | 220MB | ? | ? |
| SQLCoder-7B | 13GB | ? | ? |

**Tasks:**
1. Fill in the missing information in the table above
2. Recommend the best model for the given constraints and justify your choice
3. Explain the trade-offs between model size and performance

### Question 2 (15 points)
Design a memory optimization strategy for training on the resource-constrained system. Your answer should include:

1. **LoRA Configuration:** Specify optimal values for `r`, `alpha`, and `target_modules`
2. **Training Parameters:** Recommend `batch_size`, `gradient_accumulation_steps`, and `max_length`
3. **Memory Optimizations:** List at least 3 specific techniques to reduce memory usage
4. **File Size Management:** Explain the difference between saving adapters vs full models

---

## Part B: Implementation Tasks (40 points)

### Question 3: Training Configuration (20 points)
Write a complete `config.yaml` file for training a text-to-SQL model with the following requirements:

- Use the most memory-efficient model suitable for production
- Enable all available memory optimizations
- Configure for Mac MPS training (no CUDA)
- Set training for 1000 samples maximum
- Enable adapter-only saving
- Configure Ollama deployment with model name "company-sql-assistant"

```yaml
# Your config.yaml here
```

### Question 4: Command Line Implementation (20 points)
Write the complete command-line instructions for:

1. **Training Command:** Train the model with fast Mac optimizations using 500 samples
2. **Deployment Command:** Deploy the trained model to Ollama
3. **Testing Command:** Test the deployed model with a custom query
4. **Troubleshooting:** If deployment fails with "model not found" error, provide the fix

```bash
# Your commands here
```

---

## Part C: Technical Problem Solving (25 points)

### Question 5: Error Resolution (15 points)
You encounter the following error during training:

```
ValueError: Target modules {'k_proj', 'q_proj', 'up_proj'} not found in the base model.
```

1. **Explain** what causes this error
2. **Describe** how the system automatically resolves this issue
3. **List** alternative target modules that might work for different model architectures

### Question 6: Deployment Issues (10 points)
During Ollama deployment, you see:

```
Error: pull model manifest: file does not exist
```

1. **Identify** the root cause of this error
2. **Explain** the model name mapping process
3. **Provide** the specific solution implemented in the codebase

---

## Part D: Practical Application (10 points)

### Question 7: Real-world Usage
You need to create a text-to-SQL query for the following scenario:

**Database Schema:**
```sql
CREATE TABLE customers (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    registration_date DATE,
    city VARCHAR(50)
);

CREATE TABLE orders (
    id INT PRIMARY KEY,
    customer_id INT,
    product_name VARCHAR(100),
    amount DECIMAL(10,2),
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
```

**Natural Language Query:** "Find the top 5 customers by total order value who registered in the last 6 months and are from New York"

1. **Write the expected SQL query**
2. **Format the prompt** you would send to the Ollama model
3. **Explain** what makes this query complex and how the model should handle it

---

## Bonus Question (5 points)
**Performance Optimization:** The company wants to process 1000 queries per hour. Describe 3 specific optimizations you would implement to achieve this throughput while maintaining query quality.

---

## Evaluation Rubric

### Excellent (90-100%)
- Demonstrates deep understanding of LoRA, quantization, and memory optimization
- Provides accurate and complete configurations
- Shows practical problem-solving skills
- Explains trade-offs clearly

### Good (80-89%)
- Shows solid understanding of key concepts
- Most configurations are correct
- Can identify and solve common issues
- Minor gaps in explanation

### Satisfactory (70-79%)
- Basic understanding of the system
- Some configuration errors
- Can follow instructions but limited problem-solving
- Incomplete explanations

### Needs Improvement (<70%)
- Limited understanding of core concepts
- Multiple configuration errors
- Cannot troubleshoot issues effectively
- Poor or missing explanations

---

## Answer Key Guidelines

### Part A - Model Selection
- **Correct Choice:** CodeT5+ 770M for balanced performance and resources
- **Key Points:** Memory constraints, training time, deployment size
- **Trade-offs:** Size vs accuracy, speed vs quality

### Part B - Implementation
- **Config Requirements:** Mac optimizations, adapter saving, memory settings
- **Commands:** --fast-mac flag, proper model selection, error handling

### Part C - Problem Solving
- **LoRA Error:** Auto-detection mechanism, target module mapping
- **Deployment Error:** HuggingFace to Ollama name mapping

### Part D - Application
- **Complex Query:** Joins, aggregations, date filtering, ordering
- **Prompt Format:** Schema + natural language structure

This assessment tests both theoretical knowledge and practical implementation skills needed for real-world text-to-SQL deployment.
