from utils.utils_libs import *

'''
References:
            https://github.com/gaoliang13/FedDC/
'''

class DatasetObject:
    def __init__(self,dataset,n_client,seed,rule,unbalanced_sgm =0,rule_arg='',data_path='',held_out_domain=-1):
        '''
        Class for Dataset Preparation and Distribution among clients
        Args:
            dataset (str): the dataset to distribute (e.g. CIFAR-10, MNIST)
            n_client (int): number of clients in experiment
            seed (int): random seed
            rule (str): Distribution from which the split is sampled e.g. Drichlet
            unablanced_sgm (float): To allow for unbalanced sample counts among clients
            rule_arg (str): the value of the distribution parameter (for Drichlet, this is the alpha value)
            data_path (str): the directory to store the data splits 
        
        Methods: 
            set_data: this is called during initialization, splits and stores the data among clients

        '''
        self.dataset = dataset
        self.n_client = n_client
        self.seed = seed
        self.rule = rule
        self.rule_arg = rule_arg
        rule_arg_str = rule_arg if isinstance(rule_arg,str) else "%.3f" % rule_arg
        self.data_path = data_path
        self.unbalanced_sgm = unbalanced_sgm
        self.name = "%s_%d_%d_%s_%s" %(self.dataset, self.n_client, self.seed, self.rule, rule_arg_str)
        self.held_out_domain = held_out_domain
        self.set_data()
        
    
    def set_data(self):
        
        if not os.path.exists('%sData/%s' %(self.data_path, self.name)):
            
            if self.dataset == 'CIFAR10':
            # Load the raw data as train and test sets
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

                trnset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10
            
                trn_itr = trn_load.__iter__(); tst_itr = tst_load.__iter__() 
                
                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()

                trn_x = trn_x.numpy(); trn_y = trn_y.numpy().reshape(-1,1)
                tst_x = tst_x.numpy(); tst_y = tst_y.numpy().reshape(-1,1)
            
            elif self.dataset == 'CIFAR100':
                # Load the raw data as train and test sets
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.507, 0.487, 0.441], 
                                                                    std=[0.267, 0.256, 0.276])])

                trnset = torchvision.datasets.CIFAR100(root='%sData/Raw' % self.data_path,
                                                    train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR100(root='%sData/Raw' % self.data_path,
                                                    train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)

                self.channels = 3
                self.width = 32
                self.height = 32
                self.n_cls = 100  # CIFAR-100 has 100 classes

                trn_itr = iter(trn_load)
                tst_itr = iter(tst_load)

                trn_x, trn_y = next(trn_itr)
                tst_x, tst_y = next(tst_itr)

                trn_x = trn_x.numpy()
                trn_y = trn_y.numpy().reshape(-1, 1)
                tst_x = tst_x.numpy()
                tst_y = tst_y.numpy().reshape(-1, 1)
            
            elif self.dataset == 'PACS':
                data_path = "Data" + "/Raw/"+"PACS"
                # Define domains
                self.domains = ['art_painting', 'cartoon', 'photo', 'sketch']
                self.n_cls = 7  # PACS has 7 categories
                self.channels = 3
                self.width = 224
                self.height = 224

                # Hold one domain out (e.g., last one as default)
                held_out_domain = self.domains[self.held_out_domain]  
                # Common transforms (normalize to ImageNet if using pretrained models)
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

                domain_paths = [os.path.join(data_path, d) for d in self.domains]
                trn_x, trn_y = [], []
                tst_x, tst_y = [], []

                # Iterate over domains
                for domain, path in zip(self.domains, domain_paths):
                    dataset = ImageFolder(path, transform=transform)
                    x = [np.array(img[0].numpy()) for img in dataset]
                    y = [img[1] for img in dataset]
                    x = np.stack(x)
                    y = np.array(y).reshape(-1, 1)

                    if domain == held_out_domain:
                        tst_x.append(x)
                        tst_y.append(y)
                    else:
                        trn_x.append(x)
                        trn_y.append(y)

                trn_x = np.concatenate(trn_x, axis=0)
                trn_y = np.concatenate(trn_y, axis=0)
                tst_x = np.concatenate(tst_x, axis=0)
                tst_y = np.concatenate(tst_y, axis=0)
            
            elif self.dataset == 'OfficeHome':
                data_path = "Data" +"/Raw/"+self.dataset
                self.domains = ['Art', 'Clipart', 'Product', 'Real_World']
                self.n_cls = 65
                self.channels = 3
                self.width = 224
                self.height = 224

                held_out_domain = self.domains[self.held_out_domain]  

                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

                domain_paths = [os.path.join(self.data_path, d) for d in self.domains]
                trn_x, trn_y = [], []
                tst_x, tst_y = [], []

                for domain, path in zip(self.domains, domain_paths):
                    dataset = ImageFolder(path, transform=transform)
                    x = [np.array(img[0].numpy()) for img in dataset]
                    y = [img[1] for img in dataset]
                    x = np.stack(x)
                    y = np.array(y).reshape(-1, 1)

                    if domain == held_out_domain:
                        tst_x.append(x)
                        tst_y.append(y)
                    else:
                        trn_x.append(x)
                        trn_y.append(y)

                trn_x = np.concatenate(trn_x, axis=0)
                trn_y = np.concatenate(trn_y, axis=0)
                tst_x = np.concatenate(tst_x, axis=0)
                tst_y = np.concatenate(tst_y, axis=0)



            
            else:
                raise NotImplementedError("Invalid Dataset")
        
            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]
            
            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y
            
            ## Split Data
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            
            # Draw from lognormal distribution to create some imbalance among the clients
            print("Sampling priors from a lognormal distribution...")
            clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client))
            clnt_data_list = (clnt_data_list/np.sum(clnt_data_list)*len(trn_y)).astype(int)
            diff = np.sum(clnt_data_list) - len(trn_y)
            # print(diff)
            # Adjust the data counts for the clients to ensure that the total number of data points matches exactly.
            if diff!= 0:
                for clnt_i in range(self.n_client):
                    if clnt_data_list[clnt_i] > diff:
                        clnt_data_list[clnt_i] -= diff
                        break
            # clnt_data_list = np.ones(self.n_client,dtype=int) * n_data_per_clnt
            # print(clnt_data_list)
            # Split the dataset among multiple clients according to drichlet distribution
            print("Splitting according to Drichlet distribution with alpha=",self.rule_arg,"...")
            if self.rule == 'Drichlet' or self.rule=='Pathological':
                if self.rule == 'Drichlet':
                    cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                else:
                    c = int(self.rule_arg)
                    a = np.ones([self.n_client,self.n_cls])
                    a[:,c::] = 0
                    [np.random.shuffle(i) for i in a]
                    prior_cumsum = a.copy()
                    for i in range(prior_cumsum.shape[0]):
                        for j in range(prior_cumsum.shape[1]):
                            if prior_cumsum[i,j] != 0:
                                prior_cumsum[i,j] = a[i,0:j+1].sum()/c*1.0
                
                idx_list = [np.where(trn_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
                
                clients = np.arange(self.n_client)
                classes = np.arange(self.n_cls)
                while(np.sum(clnt_data_list)!=0):
                    curr_clnt = np.random.choice(clients)
                    # If current node is full resample a client
                    #print('Remaining Data: %d' %np.sum(clnt_data_list))
                    if clnt_data_list[curr_clnt] <= 0:
                        clients = np.setdiff1d(clients,np.array([curr_clnt]))
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt][classes]
                    while True:
                        #print(classes,np.argmax(np.random.uniform() <= curr_prior))
                        cls_label = classes[np.argmax(np.random.uniform() <= curr_prior)]
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            classes = np.setdiff1d(classes, np.array([cls_label]))
                            if len(classes) == 0:
                                break
                            continue
                        curr_prior = prior_cumsum[curr_clnt][classes]
                        cls_amount[cls_label] -= 1
                        
                        clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]

                        # print(clients,classes)
                        if cls_amount[cls_label] <= 0:
                            classes = np.setdiff1d(classes, np.array([cls_label]))
                            if len(classes) == 0:
                                break
                            curr_prior = prior_cumsum[curr_clnt][classes]
                        break
                
                for i in range(self.n_client):
                    clnt_x[i] = np.asarray(clnt_x[i])
                    clnt_y[i] = np.asarray(clnt_y[i])
                
                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt,cls] = np.mean(clnt_y[clnt]==cls)
                prior_real_diff = np.abs(cls_means-cls_priors)
                print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))

            elif self.rule == 'iid':
                
                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
            
                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                
                
                for i in range(self.n_client):
                    clnt_x[i] = np.asarray(clnt_x[i])
                    clnt_y[i] = np.asarray(clnt_y[i])

            
            self.clnt_x = clnt_x; self.clnt_y = clnt_y

            self.tst_x  = tst_x;  self.tst_y  = tst_y
            
            # Save data
            os.mkdir('%sData/%s' %(self.data_path, self.name))
            
            np.save('%sData/%s/clnt_x.npy' %(self.data_path, self.name), clnt_x, allow_pickle=True)
            np.save('%sData/%s/clnt_y.npy' %(self.data_path, self.name), clnt_y, allow_pickle=True)

            np.save('%sData/%s/tst_x.npy'  %(self.data_path, self.name),  tst_x)
            np.save('%sData/%s/tst_y.npy'  %(self.data_path, self.name),  tst_y)

        else:
            print("Data is already downloaded")
            self.clnt_x = np.load('%sData/%s/clnt_x.npy' %(self.data_path, self.name),allow_pickle=True)
            self.clnt_y = np.load('%sData/%s/clnt_y.npy' %(self.data_path, self.name),allow_pickle=True)
            self.n_client = len(self.clnt_x)

            self.tst_x  = np.load('%sData/%s/tst_x.npy'  %(self.data_path, self.name),allow_pickle=True)
            self.tst_y  = np.load('%sData/%s/tst_y.npy'  %(self.data_path, self.name),allow_pickle=True)
            
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            elif self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            elif self.dataset == "PACS" or self.dataset == "OfficeHome":
                self.domains = ['art_painting', 'cartoon', 'photo', 'sketch'] if self.dataset == "PACS" else ['Art', 'Clipart', 'Product', 'Real_World']
                self.channels = 3
                self.width = 224
                self.height = 224
                if self.dataset == "PACS":
                    self.n_cls = 7
                elif self.dataset == "OfficeHome":
                    self.n_cls = 65
                
             
        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " %clnt + 
                  ', '.join(["%.3f" %np.mean(self.clnt_y[clnt]==cls) for cls in range(self.n_cls)]) + 
                  ', Amount:%d' %self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]
        
        
        print('Total Amount:%d' %count)
        print('--------')

        print("      Test: " + 
              ', '.join(["%.3f" %np.mean(self.tst_y==cls) for cls in range(self.n_cls)]) + 
              ', Amount:%d' %self.tst_y.shape[0])
    

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
            
        if self.name == 'CIFAR10' or self.name == 'CIFAR100':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
        
            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
        
        elif self.name == 'PACS' or self.name == 'OfficeHome':
            self.train = train
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # If you want to redo transform
                transforms.Resize((224, 224)),
            ])
            self.X_data = data_x
            self.y_data = data_y.astype('int64') if not isinstance(data_y, bool) else data_y
                
        else:
            raise NotImplementedError("Not a valid dataset")
        
           
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name in ['CIFAR10', 'CIFAR100', 'PACS','OfficeHome']:
            img = self.X_data[idx]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                return img, self.y_data[idx]
        else:
            raise NotImplementedError("Not a valid dataset")