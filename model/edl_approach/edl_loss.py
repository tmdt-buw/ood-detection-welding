import torch

def one_hot_embedding(labels, num_classes=2, device=None):
        y = torch.eye(num_classes, device=device)
        return y[labels]

def get_device():
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:" + str(torch.cuda.current_device()) if use_cuda else "cpu")
        return device

def exp_evidence(y):
        return torch.exp(torch.clamp(y, -10, 10))

def kl_divergence(alpha, num_classes, device=None):
        if not device:
            device = get_device()
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            ((alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha)))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, annealing_start, device=None):
        """
        loss using the vacuity component and kl-divergence for maximizing softmax propability for id and minimize for ood data
        Paper Link: https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Aguilar_Continual_Evidential_Deep_Learning_for_Out-of-Distribution_Detection_ICCVW_2023_paper.pdf 
         Args:
            func: natural logarithmic function (ln)
            y: prediction labels
            alpha: model output logits exponentiated
            epoch_num (int): current epoch 
            num_classes (int): number of classes in the dataset
            annealing_step (int): determines how fast annealing increases (e.g. over the number of epochs)
            annealing_start (float): initial annealing parameter
        """
        y = y.to(device) 
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)#S = Dirichilet Strenght = summe der evidenzen der klasse
        
        S_0_removed = S[S != 0]
        if torch.any(S == 0):
            print("Es gibt Null-Werte in S!")

        vacuity = num_classes/S_0_removed #Laut paper: Bestimmendes Maß für die Unsischerheit, bzw score, berechenbar mit Anzahl Klassen / Drichilet Strength
        
        if torch.any(torch.isnan(vacuity)):
            print("Es gibt NaN-Werte in vacuity!")

        A = torch.sum(y * (func(S)-func(alpha)), dim=1, keepdim=True)
        # A = Tatsächlicher Loss für die Evidenz
        #func(S)-func(alpha): differenz zwischen logarithmus aus drichilet strenght und evidenz
      
        annealing_start = torch.tensor(annealing_start, dtype=torch.float32)
        annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / (annealing_step) * epoch_num)
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            annealing_coef,
        )

        kl_alpha = (alpha - 1) * (1 - y) + 1

        if epoch_num == 0:
            return A, vacuity
        kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device) 
        return (A*0.9 + kl_div*0.1), vacuity


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, annealing_start, device=None):
        """
        function for applying exponentiation on the models output and call the edl_loss function
        Paper Link: https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Aguilar_Continual_Evidential_Deep_Learning_for_Out-of-Distribution_Detection_ICCVW_2023_paper.pdf 
         Args:
            output: model output logits
            target: labels
            epoch_num (int): current epoch 
            num_classes (int): number of classes in the dataset
            annealing_step (int): determines how fast annealing increases (e.g. over the number of epochs)
            annealing_start (float): initial annealing parameter
            batch_size (int): batch size in the data 
        """
        if not device:
            device = get_device()

        alpha = torch.exp(output) + 1 #exponentierung zur generierung der evidenz für die logits

        if torch.any(torch.isnan(alpha)):
            print("Es gibt NaN-Werte in alpha!")

        y = one_hot_embedding(target, num_classes, device).float()

        loss, vacuity =  edl_loss(
                torch.log, y, alpha, epoch_num, num_classes, annealing_step, annealing_start, device=device
            )
        
        loss = torch.mean(loss)

        return loss, vacuity

