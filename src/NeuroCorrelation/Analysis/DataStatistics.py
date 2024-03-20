from src.NeuroCorrelation.Analysis.DataComparison import DataComparison

class DataStatistics():

    def __init__(self, univar_count_in, univar_count_out, dim_latent, data, path_folder):
        self.univar_count_in = univar_count_in
        self.univar_count_out = univar_count_out
        self.dim_latent = dim_latent
        self.data = data
        self.path_folder = path_folder
        self.dataComparison = DataComparison(univar_count_in=self.univar_count_in, univar_count_out=self.univar_count_out, dim_latent=self.dim_latent, path_folder=path_folder)
        self.corrCoeff = None
    
    def get_corrCoeff(self, latent):
        if self.corrCoeff is None:
            self.corrCoeff = dict()
            self.corrCoeff['input'] = self.dataComparison.correlationCoeff(self.data["inp_data_vc"])
            if latent:
                self.corrCoeff['latent']  = self.dataComparison.correlationCoeff(self.data['latent_data_bycomp']['latent'])
            self.corrCoeff['output'] = self.dataComparison.correlationCoeff(self.data["out_data_vc"])

        return self.corrCoeff

    def plot(self, plot_colors, plot_name, distribution_compare, latent=False, verbose=True, correlationCoeff=True):
        if verbose:
            print("\tPLOT: Predicted Test")
            print("\t\tdistribution analysis")
            print("\t\t\tdistribution analysis: input")
        self.dataComparison.plot_vc_analysis(self.data["inp_data_vc"], plot_name=f"{plot_name}_input", mode="in", color_data=plot_colors["input"])
        if latent:
            if verbose:
                print("\t\t\tdistribution analysis: latent")
                self.dataComparison.plot_latent_analysis(self.data['latent_data_bycomp']['latent'], plot_name=f"{plot_name}__latent", color_data=plot_colors["latent"])
        if verbose:
            print("\t\t\tdistribution analysis: output")        
        self.dataComparison.plot_vc_analysis(self.data["out_data_vc"], plot_name=f"{plot_name}_output", mode="out", color_data=plot_colors["output"])
        if verbose:
            print("\t\tdistribution analysis: real and generated")
        self.dataComparison.data_comparison_plot(distribution_compare, plot_name=f"{plot_name}", mode="out")

        if correlationCoeff:
            corrCoeff = self.get_corrCoeff(latent)
            if verbose:
                print("\t\tcorrelation analysis")
                print("\t\t\tcorrelation analysis: input")
            
            self.dataComparison.plot_vc_correlationCoeff(self.data["inp_data_vc"], plot_name=f"{plot_name}_input", corrMatrix=corrCoeff['input'])
            if latent:
                if verbose:
                    print("\t\t\tcorrelation analysis: latent")
                
                self.dataComparison.plot_vc_correlationCoeff(self.data['latent_data_bycomp']['latent'], plot_name=f"{plot_name}_latent", is_latent=True, corrMatrix=corrCoeff['latent'])
            if verbose:
                print("\t\t\tcorrelation analysis: output")
            
            self.dataComparison.plot_vc_correlationCoeff(self.data["out_data_vc"], plot_name=f"{plot_name}_output", corrMatrix=corrCoeff['output'])                 

    def draw_point_overDistribution(self, plotname, n_var, points,  distr=None):
        self.dataComparison.draw_point_overDistribution(plotname, self.path_folder, n_var, points,  distr)