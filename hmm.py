import numpy

"""Hidden Markov Model sequence tagger

"""
from classifier import Classifier

class HMM(Classifier):

    def __init__(self, model=None):
        super(Classifier, self).__init__()
        self.labels = []
        self.V = []
        self.label2index = {}
        self.index2label = []
        self.rare_words = []

    def get_model(self): return None
    def set_model(self, model): pass

    model = property(get_model, set_model)

    def get_labels(self, instance_list):
        for instance in instance_list:
            for label in instance.label:
                if label not in self.labels:
                    self.labels.append(label)

    def get_V(self, instance_list):
        for instance in instance_list:
            for t in instance.data:
                if t not in self.V and t not in self.rare_words:
                    self.V.append(t)
        self.V.append('<UNK>')

    def get_rare_words(self, instance_list):
        counts = {}
        for instance in instance_list:
            for t in instance.data:
                counts[t] = counts.get(t, 0) + 1
        for k,v in counts.items():
            if v < 3:
                self.rare_words.append(k)

    def _collect_counts(self, instance_list):
        """Collect counts necessary for fitting parameters

        This function should update self.transition_count_table
        and self.feature_count_table based on this new given instance

        Add your docstring here explaining how you implement this function

        Returns None
        """
        """ Based on each instance, I augment empirical counts for every word and its BIO label in feature_count_table and for every transition from previous label to current label in transition_count_table.
        All "rare words" (those words that appear less than 3 times) are replaced by <UNK>.
        I also add label|START counts.
        """
        # Build feature_count_table of V x labels and transition_count_table of labels x labels
        for instance in instance_list: # Set of <(w, pos), l>
            index = 0
            for t in instance.data: # Tuple of (w, pos)
                index = instance.data.index(t)
                # print t[0] # word
                # print instance.label[index] # label
                if t in self.V:
                    self.feature_count_table[self.V.index(t)][self.labels.index(instance.label[index])] +=1
                else:
                    self.feature_count_table[self.V.index('<UNK>')][self.labels.index(instance.label[index])] +=1
                if index > 0:
                    self.transition_count_table[self.labels.index(instance.label[index-1])][self.labels.index(instance.label[index])] += 1
                else:
                    self.transition_count_table[len(self.labels)][self.labels.index(instance.label[index])] += 1

    def train(self, instance_list):
        """Fit parameters for hidden markov model

        Update codebooks from the given data to be consistent with
        the probability tables

        Transition matrix and emission probability matrix
        will then be populated with the maximum likelihood estimate
        of the appropriate parameters

        Add your docstring here explaining how you implement this function

        Returns None
        """
        """Observation probabilities b_t=c(o_t=x,q_t=y)/c(q_t=y)
        Transition probabilities a_t=c(q_t-1=i,q_t=j)/c(q_t-1=i)
        Based on the empirical counts from _collect_counts, I compute probabilities for each word being emitted in given state and for each state-to-state transition, including START->state.
        <UNK> is used to account for unseen features in the training set.
        """
        # Get labels and final V (replacing rare words with <UNK>) for the training data
        self.get_labels(instance_list)
        self.get_rare_words(instance_list)
        self.get_V(instance_list)

        # Get maps of label and indices:
        for i in xrange(len(self.labels)):
            self.label2index[self.labels[i]] = i
            self.index2label.append(self.labels[i])

        # transition probabilities: matrix labels x labels
        self.transition_matrix = numpy.zeros((len(self.labels)+1,len(self.labels))) #a
        # observation probabilities: matrix of V x labels
        self.emission_matrix = numpy.zeros((len(self.V),len(self.labels))) #b
        self.transition_count_table = numpy.zeros((len(self.labels)+1,len(self.labels)))
        self.feature_count_table = numpy.zeros((len(self.V),len(self.labels)))
        self._collect_counts(instance_list)
        #TODO: estimate the parameters from the count tables
        for instance in instance_list:
            index = 0
            for t in instance.data:
                index = instance.data.index(t)
                if t in self.V:
                    self.emission_matrix[self.V.index(t)][self.labels.index(instance.label[index])] = self.feature_count_table[self.V.index(t)][self.labels.index(instance.label[index])]/self.feature_count_table[:,self.labels.index(instance.label[index])].sum()
                else:
                    self.emission_matrix[self.V.index('<UNK>')][self.labels.index(instance.label[index])] = self.feature_count_table[self.V.index('<UNK>')][self.labels.index(instance.label[index])]/self.feature_count_table[:,self.labels.index(instance.label[index])].sum()

                if index > 0:
                    self.transition_matrix[self.labels.index(instance.label[index-1])][self.labels.index(instance.label[index])] = self.transition_count_table[self.labels.index(instance.label[index-1])][self.labels.index(instance.label[index])]/self.transition_count_table[self.labels.index(instance.label[index-1]), :].sum()
                else:
                    self.transition_matrix[len(self.labels)][self.labels.index(instance.label[index])] = self.transition_count_table[len(self.labels)][self.labels.index(instance.label[index])]/self.transition_count_table[len(self.labels), :].sum()

    def classify(self, instance):
        """Viterbi decoding algorithm

        Wrapper for running the Viterbi algorithm
        We can then obtain the best sequence of labels from the backtrace pointers matrix

        Add your docstring here explaining how you implement this function

        Returns a list of labels e.g. ['B','I','O','O','B']
        """
        """best stores the pointer to the sequence with maximum final probability from the Viterbi trellis.
        Iterate through the corresponding row in the backtrace_pointers matrix backwards and add its labels to best_sequence list.
        Return the reversed best_sequence list.
        """
        backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)
        best_sequence = []
        trellis = backtrace_pointers[0]
        best = trellis[:,len(instance.data)-1].argmax()
        i = best
        best_sequence = [self.labels[best]]
        for j in xrange(len(instance.data)-1, 0, -1):
            i = backtrace_pointers[1][i][j]
            best_sequence.append(self.labels[i])
        return best_sequence[::-1]

    def dynamic_programming_on_trellis(self, instance, run_forward_alg=True):
        """Run Forward algorithm or Viterbi algorithm

        This function uses the trellis to implement dynamic
        programming algorithm for obtaining the best sequence
        of labels given the observations

        Add your docstring here explaining how you implement this function

        Returns trellis filled up with the forward probabilities
        and backtrace pointers for finding the best sequence
        """
        """Build a trellis of length of all tags +1 for tag|START x length of the instance sequence of probabilities.
        Backtrace matrix has the same dimensions but stores corresponding codes for states.
        First, I initialize the probabilities for each state at t=0 as simply emission probability*transition probability.
        backtrace_pointers at t=0 are already initialized to 0.
        Loop through time step starting from t=1 and through states and either sum up (Forward) or get max (Viterbi) of the product of the previous step and emission probability and transition probability for all states (for each another loop through states is needed).
        In Viterbi, the code(index) to the most probable state is added at the same time.
        """
        #TODO:Initialize trellis and backtrace pointers
        trellis = numpy.zeros((len(self.labels)+1,len(instance.data)))
        backtrace_pointers = numpy.zeros(shape=(len(self.labels)+1,len(instance.data)), dtype=int)
        #TODO:Traverse through the trellis here
        if run_forward_alg == True:
            for i in xrange(len(self.labels)):
                if instance.data[0] in self.V:
                    trellis[i][0] = self.transition_matrix[len(self.labels)][i]*self.emission_matrix[self.V.index(instance.data[0])][i]
                else:
                    trellis[i][0] = self.transition_matrix[len(self.labels)][i]*self.emission_matrix[self.V.index('<UNK>')][i]
            for i in xrange(len(self.labels)):
                for j in xrange(1,len(instance.data)):
                    emission_prob = 0
                    if instance.data[j] in self.V:
                        emission_prob = self.emission_matrix[self.V.index(instance.data[j])][i]
                    else:
                        emission_prob = self.emission_matrix[self.V.index('<UNK>')][i]
                    for k in xrange(1, len(self.labels)):
                        trellis[i][j] += trellis[k][j-1] * self.transition_matrix[k][i] * emission_prob
        else:
            for i in xrange(len(self.labels)):
                if instance.data[0] in self.V:
                    trellis[i][0] = self.transition_matrix[len(self.labels)][i]*self.emission_matrix[self.V.index(instance.data[0])][i]
                else:
                    trellis[i][0] = self.transition_matrix[len(self.labels)][i]*self.emission_matrix[self.V.index('<UNK>')][i]
            for j in xrange(1,len(instance.data)):
                for i in xrange(len(self.labels)):
                    emission_prob = 0
                    if instance.data[j] in self.V:
                        emission_prob = self.emission_matrix[self.V.index(instance.data[j])][i]
                    else:
                        emission_prob = self.emission_matrix[self.V.index('<UNK>')][i]
                    max_prob = trellis[0][j-1] * self.transition_matrix[0][i] * emission_prob
                    max_index = 0
                    for k in xrange(1, len(self.labels)):
                        prob = trellis[k][j-1] * self.transition_matrix[k][i] * emission_prob
                        if prob > max_prob:
                            max_prob = prob
                            max_index = k
                    trellis[i][j] = max_prob
                    backtrace_pointers[i][j] = max_index
        return (trellis, backtrace_pointers)
