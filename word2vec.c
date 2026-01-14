//word2vec from scratch

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define TABLE_SIZE 1e8

long long layer1_size = 300;
long long min_count = 50;
int window = 5;
int negative = 5;
float alpha = 0.025;
float sample = 1e-3;

struct vocab_word {
    long long cn;
    char *word;
};

struct vocab_word *vocab;
int *vocab_hash;
int *table;
float *syn0;
float *syn1neg;
float *expTable;

const int vocab_hash_size = 30000000;
long long vocab_max_size = 1000;
long long vocab_size = 0;
long long train_words = 0;
long long word_count_actual = 0;
long long min_reduce = 1;

void InitExpTable() {
    int i;
    float f;
    float numerator;
    float denominator;
    float exponent_val;
    
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    if (expTable == NULL) {
        printf("Memory allocation failed for ExpTable\n");
        exit(1);
    }

    i = 0;
    while (i < EXP_TABLE_SIZE) {
        exponent_val = (i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP;
        f = exp(exponent_val);
        numerator = f;
        denominator = f + 1;
        expTable[i] = numerator / denominator;
        i++;
    }
}

static inline float GetSigmoid(float f) {
    float result;
    int index;
    float formula_val;

    if (f > MAX_EXP) {
        return 1.0f;
    }
    if (f < -MAX_EXP) {
        return 0.0f;
    }
    
    formula_val = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
    index = (int)formula_val;
    result = expTable[index];
    
    return result;
}

static inline void RunGradientDescent(long long input_idx, long long target_idx, int label, float alpha, float *neu1_error) {
    long long l1;
    long long l2;
    float f;
    float g;
    float pred;
    float product;
    float diff;
    int c;
    
    l1 = input_idx * layer1_size;
    l2 = target_idx * layer1_size;
    f = 0;
    
    c = 0;
    while (c < layer1_size) {
        product = syn0[l1 + c] * syn1neg[l2 + c];
        f = f + product;
        c++;
    }

    pred = GetSigmoid(f);
    
    diff = label - pred;
    g = diff * alpha;

    c = 0;
    while (c < layer1_size) {
        product = g * syn1neg[l2 + c];
        neu1_error[c] = neu1_error[c] + product;
        c++;
    }

    c = 0;
    while (c < layer1_size) {
        product = g * syn0[l1 + c];
        syn1neg[l2 + c] = syn1neg[l2 + c] + product;
        c++;
    }
}

void ReadWord(char *word, FILE *fin) {
    int a;
    int ch;
    
    a = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        
        if (ch == 13) {
            continue;
        }

        if (ch == ' ') {
            if (a > 0) {
                break;
            }
            continue;
        }
        
        if (ch == '\t') {
            if (a > 0) {
                break;
            }
            continue;
        }
        
        if (ch == '\n') {
            if (a > 0) {
                ungetc(ch, fin);
                break;
            }
            strcpy(word, "</s>");
            return;
        }

        word[a] = ch;
        a++;
        
        if (a >= MAX_STRING - 1) {
            a--;
        }
    }
    word[a] = 0;
}

int GetWordHash(char *word) {
    unsigned long long a;
    unsigned long long hash;
    int len;
    
    hash = 0;
    len = strlen(word);
    
    a = 0;
    while (a < len) {
        hash = hash * 257 + word[a];
        a++;
    }
    
    hash = hash % vocab_hash_size;
    return hash;
}

int SearchVocab(char *word) {
    unsigned int hash;
    int index;
    
    hash = GetWordHash(word);
    
    while (1) {
        index = vocab_hash[hash];
        
        if (index == -1) {
            return -1;
        }
        
        if (!strcmp(word, vocab[index].word)) {
            return index;
        }
        
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

int AddWordToVocab(char *word) {
    unsigned int hash;
    int length;
    long long new_size;
    
    hash = GetWordHash(word);
    length = strlen(word) + 1;
    
    if (length > MAX_STRING) {
        length = MAX_STRING;
    }
    
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    if (vocab[vocab_size].word == NULL) {
        printf("Memory allocation failed for word string\n");
        exit(1);
    }
    
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size = vocab_max_size + 1000;
        new_size = vocab_max_size * sizeof(struct vocab_word);
        vocab = (struct vocab_word *)realloc(vocab, new_size);
        if (vocab == NULL) {
            printf("Memory allocation failed for vocab resize\n");
            exit(1);
        }
    }
    
    while (vocab_hash[hash] != -1) {
        hash = (hash + 1) % vocab_hash_size;
    }
    
    vocab_hash[hash] = vocab_size - 1;
    
    return vocab_size - 1;
}

int VocabCompare(const void *a, const void *b) {
    long long count_a;
    long long count_b;
    struct vocab_word *word_a;
    struct vocab_word *word_b;
    
    word_a = (struct vocab_word *)a;
    word_b = (struct vocab_word *)b;
    
    count_a = word_a->cn;
    count_b = word_b->cn;
    
    return count_b - count_a;
}

void SortVocab() {
    int i;
    int a;
    unsigned int hash;
    unsigned long long size;
    long long remaining_words;
    
    remaining_words = vocab_size - 1;
    if (remaining_words > 0) {
        qsort(&vocab[1], remaining_words, sizeof(struct vocab_word), VocabCompare);
    }
    
    i = 0;
    while (i < vocab_hash_size) {
        vocab_hash[i] = -1;
        i++;
    }
    
    size = vocab_size;
    train_words = 0;
    
    a = 0;
    while (a < size) {
        if ((vocab[a].cn < min_count) && (a != 0)) {
            vocab_size--;
            free(vocab[a].word);
        } else {
            hash = GetWordHash(vocab[a].word);
            
            while (vocab_hash[hash] != -1) {
                hash = (hash + 1) % vocab_hash_size;
            }
            
            vocab_hash[hash] = a;
            train_words = train_words + vocab[a].cn;
        }
        a++;
    }
    
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    if (vocab == NULL) {
        printf("Memory re-allocation failed after sorting\n");
        exit(1);
    }
}

void ReduceVocab() {
    int a;
    int b;
    unsigned int hash;
    int i;
    
    b = 0;
    a = 0;
    
    while (a < vocab_size) {
        if (vocab[a].cn > min_reduce) {
            vocab[b].cn = vocab[a].cn;
            vocab[b].word = vocab[a].word;
            b++;
        } else {
            free(vocab[a].word);
        }
        a++;
    }
    
    vocab_size = b;
    
    i = 0;
    while (i < vocab_hash_size) {
        vocab_hash[i] = -1;
        i++;
    }
    
    a = 0;
    while (a < vocab_size) {
        hash = GetWordHash(vocab[a].word);
        
        while (vocab_hash[hash] != -1) {
            hash = (hash + 1) % vocab_hash_size;
        }
        
        vocab_hash[hash] = a;
        a++;
    }
    
    min_reduce++;
}

void LearnVocabFromTrainFile() {
    char word[MAX_STRING];
    FILE *fin;
    long long a;
    long long i;
    int index;
    int new_index;
    long long threshold;
    
    a = 0;
    while (a < vocab_hash_size) {
        vocab_hash[a] = -1;
        a++;
    }
    
    fin = fopen("input.txt", "rb");
    if (fin == NULL) {
        printf("ERROR: input.txt not found!\n");
        exit(1);
    }
    
    vocab_size = 0;
    AddWordToVocab("</s>");
    
    printf("Pass 1: Reading file to build Vocabulary...\n");
    
    while (1) {
        ReadWord(word, fin);
        
        if (feof(fin)) {
            break;
        }
        
        train_words++;
        
        if ((train_words % 100000) == 0) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        
        index = SearchVocab(word);
        
        if (index == -1) {
            new_index = AddWordToVocab(word);
            vocab[new_index].cn = 1;
        } else {
            vocab[index].cn++;
        }
        
        threshold = vocab_hash_size * 0.7;
        if (vocab_size > threshold) {
            ReduceVocab();
        }
    }
    
    SortVocab();
    
    printf("\nVocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
    
    fclose(fin);
}

void InitNet() {
    long long a;
    long long b;
    unsigned long long next_random;
    long long syn0_size;
    long long syn1_size;
    float random_val;
    float normalized_val;
    
    next_random = 1;
    
    syn0_size = vocab_size * layer1_size * sizeof(float);
    syn0 = (float *)malloc(syn0_size);
    if (syn0 == NULL) {
        printf("Memory allocation failed for syn0\n");
        exit(1);
    }
    
    syn1_size = vocab_size * layer1_size * sizeof(float);
    syn1neg = (float *)malloc(syn1_size);
    if (syn1neg == NULL) {
        printf("Memory allocation failed for syn1neg\n");
        exit(1);
    }
    
    a = 0;
    while (a < vocab_size) {
        b = 0;
        while (b < layer1_size) {
            next_random = next_random * 25214903917 + 11;
            random_val = (next_random & 0xFFFF) / (float)65536;
            normalized_val = (random_val - 0.5) / layer1_size;
            syn0[a * layer1_size + b] = normalized_val;
            b++;
        }
        a++;
    }
    
    a = 0;
    while (a < vocab_size) {
        b = 0;
        while (b < layer1_size) {
            syn1neg[a * layer1_size + b] = 0;
            b++;
        }
        a++;
    }
}

void InitUnigramTable() {
    long long a;
    long long i;
    double train_words_pow;
    double d1;
    double power;
    double word_pow;
    double ratio;
    
    power = 0.75;
    train_words_pow = 0;
    
    table = (int *)malloc(TABLE_SIZE * sizeof(int));
    if (table == NULL) {
        printf("Memory allocation failed for Unigram Table\n");
        exit(1);
    }
    
    a = 0;
    while (a < vocab_size) {
        word_pow = pow(vocab[a].cn, power);
        train_words_pow = train_words_pow + word_pow;
        a++;
    }
    
    i = 0;
    d1 = pow(vocab[i].cn, power) / train_words_pow;
    
    a = 0;
    while (a < TABLE_SIZE) {
        table[a] = i;
        
        ratio = a / (double)TABLE_SIZE;
        
        if (ratio > d1) {
            i++;
            word_pow = pow(vocab[i].cn, power);
            d1 = d1 + (word_pow / train_words_pow);
        }
        
        if (i >= vocab_size) {
            i = vocab_size - 1;
        }
        
        a++;
    }
}

long long ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    int index;
    
    ReadWord(word, fin);
    
    if (feof(fin)) {
        return -1;
    }
    
    index = SearchVocab(word);
    return index;
}

//my main function(assumed)
int main() {
    int mode;
    size_t vocab_init_bytes;
    size_t hash_init_bytes;
    
    printf("========================================\n");
    printf("\n");
    printf("     Word2Vec C Engine (Teacher Ver)    \n");
        printf("\n");
    printf("========================================\n");
    printf("1. TRAIN New Model (Takes hours/days)\n");
        printf("\n");
    printf("2. PLAY with Existing Model (vectors.txt)\n");
        printf("\n");
    
    printf("Select Mode: ");
    
    scanf("%d", &mode);

    if (mode == 1) {
        vocab_init_bytes = vocab_max_size * sizeof(struct vocab_word);
        vocab = (struct vocab_word *)calloc(1, vocab_init_bytes);
        if (vocab == NULL) {
            printf("Initial vocab allocation failed\n");
            printf("\n");
            exit(1);
        }
        
        hash_init_bytes = vocab_hash_size * sizeof(int);
        vocab_hash = (int *)calloc(1, hash_init_bytes);
        if (vocab_hash == NULL) {
            printf("Initial hash allocation failed\n");
            exit(1);
        }
        
        InitExpTable();
        LearnVocabFromTrainFile();
        InitNet();
        InitUnigramTable();
        TrainModel();
    } 
    
    if (mode == 2) {
        LoadModel();
        InteractiveLoop();
    }
    
    return 0;
}
