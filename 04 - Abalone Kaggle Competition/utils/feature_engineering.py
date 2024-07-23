def create_features(df):
    from numpy import nan, inf

    # Normalizing constants
    max_length = df['Length'].max()
    max_diameter = df['Diameter'].max()
    max_height = df['Height'].max()
    max_whole_weight = df['Whole weight'].max()
    max_shell_weight = df['Shell weight'].max()
    
    # Creating new features
    df['Weight_Length_Ratio'] = df['Whole weight'] / df['Length']
    df['Weight_Diameter_Ratio'] = df['Whole weight'] / df['Diameter']
    df['Weight_Height_Ratio'] = df['Whole weight'] / df['Height']
    df['Shell_Weight_Ratio'] = df['Shell weight'] / df['Whole weight']
    
    df['Volume'] = df['Length'] * df['Diameter'] * df['Height']
    
    df['Normalized_Length'] = df['Length'] / max_length
    df['Normalized_Diameter'] = df['Diameter'] / max_diameter
    df['Normalized_Height'] = df['Height'] / max_height
    df['Normalized_Whole_Weight'] = df['Whole weight'] / max_whole_weight
    df['Normalized_Shell_Weight'] = df['Shell weight'] / max_shell_weight
    
    df['Density'] = df['Whole weight'] / df['Volume']
    
    df['Length_Diameter_Interaction'] = df['Length'] * df['Diameter']
    df['Length_Height_Interaction'] = df['Length'] * df['Height']
    df['Diameter_Height_Interaction'] = df['Diameter'] * df['Height']
    
    df['Total_Weight'] = df['Whole weight'] + df['Whole weight.1'] + df['Whole weight.2']
    df['Weight_Difference'] = df['Whole weight'] - df['Shell weight']

    df = df.replace(to_replace= inf, value= nan)
    
    return df