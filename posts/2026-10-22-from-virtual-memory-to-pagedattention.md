# From Virtual Memory to PagedAttention: How Block Tables Map KV Cache

A 7B model burns ~512 KB of HBM per generated token. On an A100 with 40 GB, that's 80k tokens total — yet serving a batch of 100 concurrent requests wastes more than half of it. The culprit isn't the math; it's how the KV cache is laid out.

The first question to be solved is the definition of KV cache. When LLM is inferring, key and value must be saved for memory. That's why we need cache for K and V, which is called as KV cache.

The contiguous allocation brings severe problems. As allocation size is fixed, internal fragments appear if usage size is shorter than allocation size. What's more, after several times of allocation, new request may be refused if leaving space is enough but be divided into different places, which we called this phenomenon as external fragments.

![KV cache fragmentation](https://aka.doubaocdn.com/s/lI2BjDnzJl)

Now let's review what OS does to solve the problem. It uses a technology called paging. Paging divides virtual memory into pages and divides main memory into page frames at the same size, therefore the precise mapping relationship that is recorded at files named page tables can be established. Each process has its own page table. What's more, pages are continuous while page frame could be discrete, so external fragments would not appear.

In LLM inference, we face exactly the same situation. Before PagedAttention, the KV cache of each request was allocated as one continuous block whose size was fixed by the maximum sequence length, even if the request only ended up using a small part of it. vLLM borrowed the paging idea and applied it to the KV cache. That is PagedAttention.

The mapping between OS paging and PagedAttention is almost one to one:

| OS virtual memory | PagedAttention |
|---|---|
| Page (fixed size) | KV block (fixed number of tokens, e.g. 16) |
| Page table | Block table |
| Virtual page number | Logical block number |
| Physical page frame | Physical KV block in GPU memory |
| Process | Sequence (one request) |
| External fragmentation | Eliminated |

The rule is simple. The KV cache of one sequence is not stored continuously. It is cut into fixed-size blocks. Each block holds the K and V of 16 tokens, for all layers. For a 7B model, one token takes about 512 KB of KV cache, so one block of 16 tokens is about 8 MB.

The blocks do not have to be adjacent in GPU memory. A global block pool manages all free physical blocks. Each sequence keeps its own block table: the i-th row tells us which physical block stores the i-th logical block of this sequence. Just like each process has its own page table, each sequence has its own block table.

One thing is different from OS. OS page tables are multi-level, with TLB cache and page faults. The block table here is just a flat array, and physical blocks are taken from a pre-allocated pool, so there is no page fault to handle. The idea is the same, but the implementation is much simpler.

## A concrete example

Now let us walk through one concrete example. Suppose the block size is 16 tokens, and a request has generated 37 tokens so far.

37 tokens take ceil(37 / 16) = 3 logical blocks:

- Logical block 0 holds token 0 to 15, fully used.
- Logical block 1 holds token 16 to 31, fully used.
- Logical block 2 holds token 32 to 36, only 5 slots used out of 16.

The block table of this sequence may look like this:

| Logical block | Physical block |
|---|---|
| 0 | #7 |
| 1 | #3 |
| 2 | #12 |

#7, #3 and #12 are three physical blocks picked from the global pool. They are not adjacent in memory, and nobody cares. When token 37 comes, the attention kernel first reads the block table: logical block 2 still has free slots, so it writes the new K and V into physical block #12 at slot 5. When block 2 is full (16 tokens), the next allocation just grabs a new free physical block from the pool and appends its number to the block table. No copying, no compaction, no need to move old data.

Compare this with the contiguous way. In the old way, we reserved one continuous block for the maximum length, say 2048 tokens, even though only 37 are used. Most of the space is wasted before the request even finishes. In the paged way, at most 15 slots (block size minus one) are wasted, and only in the last partial block.

## What it buys us

So what does this buy us?

First, internal fragmentation drops a lot. In the contiguous scheme, a request that ends at 500 tokens still paid for 2048. In the paged scheme, the waste is at most block_size - 1 = 15 tokens per sequence, in the last block only. vLLM reports the waste is below 4% in typical serving workloads, compared with 60% to 80% in the old way.

Second, external fragmentation is gone, because all free blocks form one pool and any sequence can take any free block.

What is more, two features come for free from the paging idea. When several requests share the same prompt prefix, their blocks can point to the same physical block, just like shared memory between processes. And when GPU memory runs out, some blocks can be swapped to CPU memory and loaded back later, just like swap space in OS.

## What it doesn't buy

But PagedAttention does not solve everything. It fixes the layout of the KV cache, not the attention itself.

The attention kernel now has to read the block table first, and then gather K and V from scattered physical blocks. This extra indirection has a cost. And decode is still memory bound: for each new token, the model still has to read the whole K and V history of this sequence from HBM. PagedAttention lets us fit more sequences in the same GPU, but each single decode step is still limited by HBM bandwidth.

That is the core idea of PagedAttention. Next time we will look at why decode is memory bound, and how prefill and decode are actually two different workloads.
