import numpy as np
import random
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit
from scipy.linalg import eig, eigh
from scipy.stats import unitary_group


## THE UNITARY FUNCTIONS

# unitaries
def U_random_layer(N, offset):
  N_gate = (N-offset)//2
  U_gates = block_diag( *([unitary_group.rvs(2) for _ in range(N_gate)]) )

  if offset == 0 and N%2==0:
      return U_gates
  elif offset == 1 and N%2==0:
      return block_diag(1.,U_gates,1.)
  elif offset == 0 and N%2==1:
      return block_diag(U_gates,1.)
  elif offset == 1 and N%2==1:
      return block_diag(1.,U_gates)

def U_layered(N,D=2, offset=None):
  if offset is None:
    offset = random.randint(0, 1)
  U = np.eye(N)
  for i in range(D):
    U_l = U_random_layer(N, (offset+i) % 2)
    U = U_l @ U
  return U

def U_haar(N):
  return np.array( unitary_group.rvs(N) )
  
  
  
## THE TRAJECTORY FUNCTIONS

# apply jump and return new G
def apply_jump(G, i, jump="create"):

  ni = G[i,i]
  if jump=="annihilate":
    return G - np.outer(G[:,i], G[i,:])/ni
  else:
    
    id_m_Gmi = -np.asarray(G[:,i]).squeeze()
    id_m_Gin = -np.asarray(G[i,:]).squeeze()
  
    id_m_Gmi[i] += 1.
    id_m_Gin[i] += 1.

    return G + np.outer(id_m_Gmi,id_m_Gin)/(1.-ni)

# get bipartite entropy of subsystem size l of adjacent sites
def get_entropy(G,l, eps=1e-10):
  N = G.shape[0]

  if l<=0 or l>=N:
    return 0.

  if l <= N//2:
    sG = G[:l,:l]
  else:
    sG = G[l:,l:]
  es = eigh(sG)[0]
  return -np.real(np.sum([x*np.log(x) + (1.-x)*np.log(1.-x) for x in es if x>eps and x<1.-eps ]) )

# more for a check, compute bipartite entropy between l and N-l sites, but don't optimize for min(l,N-l)
def get_entropy_full(G,l, eps=1e-10):
  N = G.shape[0]

  if l<=0:
    return 0.

  sG = G[:l,:l]
  es = eigh(sG)[0]  # for renormalization
  return -np.real(np.sum([x*np.log(x) + (1.-x)*np.log(1.-x) for x in es if x>0.+eps and x<1.-eps ]) )

# normalize G by imposing that eigenvalues should be 0 or 1 for pure state
def normalize(G):
  es, V = eigh(G)
  return V @ np.diag(np.real(np.round(es))) @ V.T.conj()

def sample_trajectory(N, U_func, p, N_sample=100, clicks_per_sample=100, G_start=None, it_normalize=1, U_const=False, track=False, verbose=False):

  # set start state (vacuum if None)
  if G_start is None:
    G_start = np.zeros((N,N))
  G = G_start

  if U_const:
    U = U_func()

  # number of excitations to start from
  N_exc = np.trace(G_start)

  # loop over all clicks and save data in dictionary: entanglement, G(t), and clicks observed
  data = {"S":[], "G":[], "clicks":[]}

  if track:
    bar = progressbar.ProgressBar(maxval=N_click-1, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
  for c in range(N_sample*clicks_per_sample):

    do_annihilate = (N_exc / N) > random.random()
    do_direct = p > random.random()

    # check if G is in the basis to apply the jump, otherwise transform
    if not do_direct: # sample U if not const and apply transform
      if not U_const:
        U = U_func(N)
      G = U.T.conj() @ G @ U
    
    n_p = np.maximum(0.,np.minimum(1.,np.real(np.diag(G))))
    if do_annihilate:
      i_click = random.choices(range(N), weights=np.real(n_p/N_exc ))[0]
      G = apply_jump(G, i_click, jump='annihilate')
      N_exc -= 1
    else:
      i_click = random.choices(range(N), weights=np.real((1.-n_p)/(N-N_exc)))[0]
      G = apply_jump(G, i_click, jump='create')
      N_exc += 1

    # transform back
    if not do_direct:
      G = U @ G @ U.T.conj()

    # normalize if required
    if (c+1) % it_normalize == 0:
      G = normalize(G)

    if verbose:
      print("N_exc={}".format(np.trace(G)))

    # save data
    if (c+1) % clicks_per_sample == 0:

      data['G'].append( G )
      data['S'].append( [get_entropy(G,l) for l in range(N+1) ] )
    
    # save click
    data["clicks"].append("{} {} {}".format("c" if do_annihilate else "cd",
                                            "d" if do_direct else "U",
                                            i_click))
    
    if track:
      bar.update(c)

  if track:
    bar.finish()

  return data
