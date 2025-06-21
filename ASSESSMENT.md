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
| CodeLlama-7B | 13GB | 2-3 hours | Large-scale SQL generation |
| CodeT5+ 770M | 770MB | 30-45 min | Balanced performance/efficiency |
| CodeT5+ 220M | 220MB | 15-20 min | Ultra-fast prototyping |
| SQLCoder-7B | 13GB | 2-4 hours | Specialized SQL tasks |
| **CodeBERT-base** | **~500MB** | **20-30 min** | **Quick experimentation** |

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
Write a complete training command for the following scenario:

**Scenario:** You need to quickly test the training pipeline using Microsoft's CodeBERT-base model on a Mac with Apple Silicon. You want to use aggressive optimizations for the fastest possible training with a small dataset.

**Requirements:**
- Use Microsoft CodeBERT-base as the foundation model
- Enable aggressive Mac optimizations for fastest training
- Run training only (not full pipeline)
- Use the most memory-efficient settings possible

**Your Task:**
1. Write the complete command-line instruction
2. Explain what each flag does and why you chose it
3. Predict the expected training time and memory usage
4. Identify any potential issues with this configuration

**Example Command Structure:**
```bash
python text_to_sql_train.py [YOUR FLAGS HERE]
```

**Expected Command:**
```bash
python text_to_sql_train.py --fast-mac --base-model microsoft/CodeBERT-base --mode train
```

**Flag Explanations Required:**
- `--fast-mac`: Purpose and specific optimizations applied
- `--base-model microsoft/CodeBERT-base`: Model choice justification
- `--mode train`: Why training-only vs full pipeline

**Additional Considerations:**
- Discuss CodeBERT vs CodeLlama for text-to-SQL tasks
- Explain Mac MPS optimization benefits and limitations
- Predict resource usage and training time
- Enable adapter-only saving
- Configure Ollama deployment with model name "company-sql-assistant"

```yaml
# Your config.yaml here
```

### Question 4: Command Line Implementation (20 points)
Write the complete command-line instructions for the following scenarios:

1. **CodeBERT Training Command:** Train using Microsoft CodeBERT-base with aggressive Mac optimizations
   ```bash
   python text_to_sql_train.py --fast-mac --base-model microsoft/CodeBERT-base --mode train
   ```

2. **Production Training Command:** Train a production-ready model using 2000 samples
3. **Deployment Command:** Deploy the trained CodeBERT model to Ollama with custom name "codebert-sql"
4. **Testing Command:** Test the deployed model with CPU-only mode

**Required Analysis:**
- Explain each flag in the CodeBERT command
- Compare expected results vs default CodeLlama training
- Predict memory usage and training time
- Identify potential compatibility issues

**Additional Commands to Complete:**
```bash
# Production training (fill in flags):
python text_to_sql_train.py [YOUR FLAGS]

# Deployment with custom name (fill in flags):
python text_to_sql_train.py [YOUR FLAGS]

# CPU-only testing (fill in flags):
python text_to_sql_train.py [YOUR FLAGS]
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

### Question 6: Deployment Issues (15 points)
During Ollama deployment, you encounter these issues:

**Issue 1:** 
```
Error: pull model manifest: file does not exist
```

**Issue 2:**
When testing the model, you get repeating output:
```
{ end }<|end|>
{ end }<|end|>
{ end }<|end|>
...
```

1. **Identify** the root cause of each error
2. **Explain** the model name mapping process for Issue 1
3. **Describe** what causes the token generation loop in Issue 2
4. **Provide** the specific solutions implemented in the codebase for both issues

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
- **CodeBERT Alternative:** CodeBERT-base for rapid prototyping and testing
- **Key Points:** Memory constraints, training time, deployment size
- **Trade-offs:** Size vs accuracy, speed vs quality

### Part B - Implementation
- **CodeBERT Config:** `--fast-mac --base-model microsoft/CodeBERT-base --mode train`
- **Mac Optimizations:** Batch size 2, max length 128, single epoch
- **Expected Performance:** 20-30 min training, ~2GB memory usage
- **Commands:** Proper flag usage, model selection rationale

### CodeBERT-Specific Assessment Points:
- **Model Size:** ~500MB vs 13GB CodeLlama (efficient for testing)
- **Training Speed:** 20-30 minutes with fast-mac optimizations
- **Memory Usage:** ~2GB RAM vs 8-16GB for larger models
- **Use Case:** Excellent for prototyping, limited for complex SQL
- **Mac Compatibility:** Native MPS support, no quantization issues

### Part C - Problem Solving
- **LoRA Error:** Auto-detection mechanism, target module mapping
- **Deployment Error:** HuggingFace to Ollama name mapping

### Part D - Application
- **Complex Query:** Joins, aggregations, date filtering, ordering
- **Prompt Format:** Schema + natural language structure

This assessment tests both theoretical knowledge and practical implementation skills needed for real-world text-to-SQL deployment.

---

## Practical Assessment: CodeBERT Training Execution

### Real-World Scenario: Quick Model Validation

**Command to Execute:**
```bash
python text_to_sql_train.py --fast-mac --base-model microsoft/CodeBERT-base --mode train
```

### Expected Outcomes & Assessment Criteria

#### 1. **Training Performance Metrics (20 points)**
Students should observe and report:
- **Memory Usage:** ~1.5-2.5GB peak memory consumption
- **Training Time:** 15-25 minutes on Apple Silicon Mac
- **Model Size:** Final adapter files ~50-100MB vs full model ~500MB
- **Sample Processing:** ~500 samples in single epoch

#### 2. **Configuration Validation (15 points)**
Verify students understand these auto-applied settings:
```yaml
# Auto-applied with --fast-mac
batch_size: 2
max_length: 128
num_epochs: 1
max_samples: 500
gradient_accumulation_steps: 8
use_gradient_checkpointing: False
```

#### 3. **Model Comparison Analysis (15 points)**
Students should compare CodeBERT-base vs default CodeLlama:

| Metric | CodeBERT-base | CodeLlama-7B |
|--------|---------------|--------------|
| Base Size | 500MB | 13GB |
| Training Time | 20 min | 2-3 hours |
| Memory Usage | 2GB | 8-16GB |
| SQL Quality | Good for simple | Excellent |
| Production Ready | Testing only | Yes |

#### 4. **Troubleshooting Skills (10 points)**
Common issues students might encounter:

**Issue 1:** Model loading failure
```
Solution: CodeBERT uses different architecture - auto-detection handles this
```

**Issue 2:** MPS compatibility
```
Solution: --fast-mac automatically disables problematic quantization
```

**Issue 3:** Adapter saving
```
Expected: Only LoRA adapters saved (~50MB) not full model
```

### Grading Rubric for CodeBERT Assessment

#### **Excellent (90-100%)**
- Successfully executes command and explains all optimizations
- Accurately predicts and measures performance metrics
- Clearly articulates CodeBERT vs CodeLlama trade-offs
- Demonstrates understanding of Mac MPS optimizations

#### **Good (80-89%)**
- Executes command successfully with minor explanation gaps
- Most performance predictions accurate
- Understands basic model differences
- Can identify optimization benefits

#### **Satisfactory (70-79%)**
- Command execution successful but limited analysis
- Some performance understanding
- Basic awareness of model differences
- Follows instructions without deep insight

#### **Needs Improvement (<70%)**
- Command execution issues or failure to explain
- Poor performance prediction or measurement
- Limited understanding of optimizations
- Cannot articulate model selection rationale

### Key Learning Objectives Assessed

1. **Practical CLI Usage:** Proper flag usage and command construction
2. **Resource Management:** Understanding memory and time constraints
3. **Model Selection:** Choosing appropriate models for specific use cases
4. **Optimization Understanding:** Mac-specific and fast training concepts
5. **Performance Analysis:** Measuring and interpreting training results

This practical assessment validates both theoretical knowledge and hands-on implementation skills essential for production ML deployment.
