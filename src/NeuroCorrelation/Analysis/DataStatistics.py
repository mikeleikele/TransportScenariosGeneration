from src.NeuroCorrelation.Analysis.DataComparison import DataComparison
from termcolor import cprint
from colorama import init, Style

class DataStatistics():

    def __init__(self,  univar_count_in, univar_count_out, latent_dim, data, path_folder, model_type, key_value_list, name_key="ae"):
        self.univar_count_in = univar_count_in
        self.univar_count_out = univar_count_out
        self.latent_dim = latent_dim
        self.key_value_list = key_value_list
        self.data = data
        self.name_key=name_key
        self.path_folder = path_folder
        self.dataComparison = DataComparison(key_value_list=self.key_value_list, univar_count_in=self.univar_count_in, univar_count_out=self.univar_count_out, latent_dim=self.latent_dim, path_folder=path_folder, name_key=self.name_key)
        self.corrCoeff = None
        if model_type in ["VAE"]:
            self.latent_keys = ["mu","logvar"]
        else:
            self.latent_keys = ["latent"]
            
    def get_corrCoeff(self, latent):
        if self.corrCoeff is None:
            self.corrCoeff = dict()
            self.corrCoeff['input'] = self.dataComparison.correlationCoeff(self.data["inp_data_vc"])
            if latent:
                self.corrCoeff['latent']  = dict()
                for latent_key in self.latent_keys:
                    self.dataComparison.correlationCoeff(self.data['latent_data_bycomp']['latent'][latent_key])
            self.corrCoeff['output'] = self.dataComparison.correlationCoeff(self.data["out_data_vc"])

        return self.corrCoeff

    def plot(self, plot_colors, plot_name, distribution_compare, latent=False, verbose=True, draw_correlationCoeff=False):
        if verbose:
            cprint(Style.BRIGHT +"| \t\tdistribution analysis", 'green')
            cprint(Style.BRIGHT +"| \t\t\tinput", 'green')

    
        if draw_correlationCoeff:
            self.dataComparison.plot_vc_analysis(self.data["inp_data_vc"], plot_name=f"{plot_name}_input", mode="in", color_data=plot_colors["input"])
        if latent:
            if verbose:
                cprint(Style.BRIGHT +"| \t\t\tlatent", 'green')
            for latent_key in self.latent_keys:

                self.dataComparison.plot_latent_analysis(self.data['latent_data_bycomp'][latent_key], plot_name=f"{plot_name}_latent_{latent_key}", color_data=plot_colors["latent"])
            
            if draw_correlationCoeff:
                self.dataComparison.plot_latent_corr__analysis(self.data['latent_data_bycomp'][latent_key], plot_name=f"{plot_name}_latent_{latent_key}", color_data=plot_colors["latent"])
                    
        if verbose:
            cprint(Style.BRIGHT +"| \t\t\toutput", 'green')      
        if draw_correlationCoeff:
            self.dataComparison.plot_vc_analysis(self.data["out_data_vc"], plot_name=f"{plot_name}_output", mode="out", color_data=plot_colors["output"])

        
        if verbose:
            cprint(Style.BRIGHT +"| \t\tdistribution analysis: real and generated", 'green')
        self.dataComparison.data_comparison_plot(distribution_compare, plot_name=f"{plot_name}", mode="out")
        
        
        if draw_correlationCoeff:
            corrCoeff = self.get_corrCoeff(latent)
            if verbose:
                cprint(Style.BRIGHT +"| \t\tdistribution analysis: real and generated", 'green')
                cprint(Style.BRIGHT +"| \t\t\tinput", 'green')
            
            self.dataComparison.plot_vc_correlationCoeff(self.data["inp_data_vc"], plot_name=f"{plot_name}_input", corrMatrix=corrCoeff['input'])
            if latent:
                if verbose:
                    cprint(Style.BRIGHT +"| \t\t\tlatent", 'green')
                for latent_key in self.latent_keys:
                    self.dataComparison.plot_vc_correlationCoeff(self.data['latent_data_bycomp']['latent'][latent_key], plot_name=f"{plot_name}_latent_{latent_key}", is_latent=True, corrMatrix=corrCoeff['latent'])
            if verbose:
                cprint(Style.BRIGHT +"| \t\t\toutput", 'green')
            self.dataComparison.plot_vc_correlationCoeff(self.data["out_data_vc"], plot_name=f"{plot_name}_output", corrMatrix=corrCoeff['output'])                 

    def draw_point_overDistribution(self, plotname, n_var, points,  distr=None):
        self.dataComparison.draw_point_overDistribution(plotname, self.path_folder, n_var, points,  distr)