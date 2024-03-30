from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np
from simcse import SimCSE
from sklearn.metrics.pairwise import cosine_similarity
from data_utils.utils import strip_computations

from openai.embeddings_utils import get_embedding
import openai, time

class OpenAIEmbedder:
    def __init__(self, model='text-embedding-ada-002'):
        self.model = model
    def encode(self, text_list):
        response = openai.Embedding.create(input=text_list, model=self.model)
        all_embs = np.array([response['data'][i]["embedding"] for i in range(len(text_list))])
        assert len(all_embs) == len(text_list)
        time.sleep(0.1)
        return all_embs

class Aligner:
    def get_minimum_penalty(self, x, y, sim):
        """
        Function to find out the minimum penalty to align two trajectories
        x: list of trajectory 1 steps
        y: list of trajectory 2 steps
        :param sim: similarity matrix (m x n) between two trajectories steps
        """
        pgap = self.gap_cost
        sim_threshold = self.sim_threshold

        # initializing variables
        i = 0
        j = 0

        # pattern lengths
        m, n = sim.shape
        assert len(x) == m
        assert len(y) == n

        # table for storing optimal substructure answers
        dp = np.zeros([m+1,n+1], dtype=float) #int dp[m+1][n+1] = {0};

        # initialising the table
        dp[0:(m+1),0] = [ i * pgap for i in range(m+1)]
        dp[0,0:(n+1)] = [ i * pgap for i in range(n+1)]

        # calculating the minimum penalty
        i = 1
        while i <= m:
            j = 1
            while j <= n:
                if sim[i - 1][j - 1] >= sim_threshold:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1] + 1 - sim[i - 1][j - 1],
                                    dp[i - 1][j] + pgap,
                                    dp[i][j - 1] + pgap)
                j += 1
            i += 1

        cost_of_alignment = dp[m][n]
        # Reconstructing the solution
        l = n + m # maximum possible length
        i = m
        j = n

        xpos = l
        ypos = l

        # Final answers for the respective strings
        xans = ["" for _ in range(l+1)]
        yans = ["" for _ in range(l+1)]

        while not (i == 0 or j == 0):
            #print(f"i: {i}, j: {j}")
            if sim[i - 1][j - 1] >= sim_threshold:	
                xans[xpos] = x[i - 1]
                yans[ypos] = y[j - 1]
                xpos -= 1
                ypos -= 1
                i -= 1
                j -= 1
            elif np.isclose(dp[i - 1][j - 1] + 1 - sim[i-1][j-1], dp[i][j]):
                xans[xpos] = x[i - 1]
                yans[ypos] = y[j - 1]
                xpos -= 1
                ypos -= 1
                i -= 1
                j -= 1
            
            elif np.isclose(dp[i - 1][j] + pgap, dp[i][j]):
                xans[xpos] = x[i - 1]
                yans[ypos] = '_'
                xpos -= 1
                ypos -= 1
                i -= 1
            
            elif np.isclose(dp[i][j - 1] + pgap, dp[i][j]):	
                xans[xpos] = '_'
                yans[ypos] = y[j - 1]
                xpos -= 1
                ypos -= 1
                j -= 1
            

        while xpos > 0:
            if i > 0:
                i -= 1
                xans[xpos] = x[i]
                xpos -= 1
            else:
                xans[xpos] = '_'
                xpos -= 1

        while ypos > 0:
            if j > 0:
                j -= 1
                yans[ypos] = y[j]
                ypos -= 1
            else:
                yans[ypos] = '_'
                ypos -= 1

        # Since we have assumed the answer to be n+m long,
        # we need to remove the extra gaps in the starting
        # id represents the index from which the arrays
        # xans, yans are useful
        id = 1
        i = l
        while i >= 1:
            if (yans[i] == '_') and xans[i] == '_':
                id = i + 1
                break
            
            i -= 1

        i = id
        x_seq = []
        while i <= l:
            x_seq.append(xans[i])
            i += 1
        #print(f"X seq: {x_seq}")

        # Y
        i = id
        y_seq = []
        while i <= l:
            y_seq.append(yans[i])
            i += 1
        #print(f"Y seq: {y_seq}")

        return x_seq, y_seq, cost_of_alignment

class StepAligner(Aligner):
    def __init__(self, model='roscoe', gap_cost=0.5, sim_threshold=0.8):
        self.gap_cost = gap_cost
        self.sim_threshold = sim_threshold
        if 'roscoe' in model:
            self.model = SimCSE('facebook/roscoe-512-roberta-base')
        elif 'simcse' in model:
            self.model = SimCSE('princeton-nlp/sup-simcse-roberta-base')
        elif 'openai' in model:
            self.model = OpenAIEmbedder()
        elif 'mpnet' in model:
            self.model = SimCSE('sentence-transformers/stsb-mpnet-base-v2')
        else:
            raise ValueError('Aligner Model {} not supported'.format(model))

    def compute_alignment_from_trajectories(self, trajectory1, trajectory2, delimiter=' |'):
        ## return a list of length len(trajectory1) where each element is the index of the most similar sentence in trajectory2
        ## strip computations from trajectory1 and trajectory2 before computing similarity
        
        trajectory1_noc = [strip_computations(s) for s in trajectory1]
        trajectory2_noc = [strip_computations(s) for s in trajectory2]

        ## remove delimiter from the end of each step in trajectory1 and trajectory2

        trajectory1_noc = [s[:-len(delimiter)].strip() if s.endswith(delimiter) else s for s in trajectory1_noc]
        trajectory2_noc = [s[:-len(delimiter)].strip() if s.endswith(delimiter) else s for s in trajectory2_noc]
        
        scores = self.compute_similarity(trajectory1_noc, trajectory2_noc)
        ## match each sentence in trajectory1 to the most similar sentence in trajectory2
        aligned_traj1, aligned_traj2, cost = self.get_minimum_penalty(trajectory1, trajectory2, scores)
        return aligned_traj1, aligned_traj2, cost

    def compute_similarity(self, trajectory1, trajectory2):
        all_embs = self.model.encode(trajectory1 + trajectory2)
        traj1_embs = all_embs[:len(trajectory1)]
        traj2_embs = all_embs[len(trajectory1):]   
        scores = cosine_similarity(traj1_embs, traj2_embs)
        assert scores.shape == (len(trajectory1), len(trajectory2))
        return scores
        

if __name__ == '__main__':
    #aligner = StepAligner(model='simcse', sim_threshold=0.8, gap_cost=0.5)
    aligner = StepAligner(model='mpnet', sim_threshold=0.8, gap_cost=0.5)

    sol1 = """First find how many gallons of sand Bill puts in each jug: 2 gallons * 70% = << 2*70*.01=1.4 >> 1.4 gallons | Then multiply that number by the weight per gallon to find the total weight: 1.4 gallons * 5 pounds/gallon = << 1.4*5=11 >> 11 pounds |"""
    
    sol2 = """
    # First find the volume of the sand in one jug: 2 gallons * 70% = <<2*70*.01=1.4>>1.4 gallons |
    #Then double this amount since there are two jugs: 1.4 gallons * 2 = <<1.4*2=2.8>>2.8 gallons |
    #Then multiply the total volume by the density of sand to find the total weight: 2.8 gallons * 5 pounds/gallon = <<2.8*5=14>>14 pounds |"""

    trajectory1 = [s.strip() for s in sol1.split(' |') if s.strip()]
    trajectory2 = [s.strip() for s in sol2.split(' |') if s.strip()]

    #trajectory1 = ['n0 = 70.0 ;', 'n1 = 10.0 ;', 't0 = 100.0 - n1 ;', 't1 = t0 / n0 ;', 't2 = n1 * t1 ;', 't3 = t2 - 1.0 ;', 'ans = t3 * 100.0 ;']
    #trajectory2 = ['n0 = 70.0 ;', 'n1 = 10.0 ;', 't0 = 100.0 - n1 ;', 't1 = t0 / n0 ;', 't2 = t1 - 1.0 ;', 'answer = t2 * 100.0 ;']
    #trajectory1 = ['I am a student', 'I am a teacher', 'I am a professor', 'I am a researcher']
    #trajectory2 = ['I am a student', 'I am a researcher', 'I am a teacher', 'I am a professor']
    atraj1, atraj2, cost = aligner.compute_alignment_from_trajectories(trajectory1, trajectory2, delimiter=' ;')
    print(cost)
    
    for a,b in zip(atraj1, atraj2):
        print("*****")
        print("{} ----> {}".format(a, b))
