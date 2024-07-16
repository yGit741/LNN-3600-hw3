r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
student_name_1 = 'Yonatan Greenshpan' # string
student_ID_1 = '204266191' # string
student_name_2 = 'David Bendayan' # string
student_ID_2 = '033530700' # string

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**  
There are several reason why we prefer to train our RNN on sequences and not on the whole text. First, we have memory constraints and feeding the entire corpus at once can exceed the available memory. In addition, RNN has difficulties handling long dependencies and spliting into chunk help model to focus on more managable chunks. finally, the sequencing allow us using optimization technique like mini-batch  gradient descent and truncated backpropagation.
"""

part1_q2 = r"""
**Your answer:**  
In our RNN, we carried over the hidden state from one sequence to the next sequence during the training, and reset the state only by the end of each epoch. For example: we tarin on sequence 1 and the initial state is $h_0$ and the output is $h_1$ and in sequence 2 the initial hidden state is $h_1$ and the output is $h_2$ - as we can see we have the overlapping of $h_1$ that in a transitive manner crate kind of long term memory.   
 
"""

part1_q3 = r"""
**Your answer:**  
Not shuffling the order of batches when training RNNs allows for the hidden state to carry over from one sequence to the next within an epoch, preserving context and continuity essential for learning long-term dependencies in the data.
"""

part1_q4 = r"""
**Your answer:**  
1. Lowering the temperature makes the model's predictions more confident by sharpening the probability distribution, leading to more coherent and sensible text generation.  
2. When the temperature is very high, the probability distribution becomes more uniform, resulting in more random and less coherent text because the model is less certain about its predictions.  
3. When the temperature is very low, the probability distribution becomes peaked, making the model highly confident in its predictions. This can lead to repetitive and deterministic text as the model consistently chooses the highest probability options.  

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
