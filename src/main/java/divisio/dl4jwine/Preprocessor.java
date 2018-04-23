package divisio.dl4jwine;

import java.io.File;

public interface Preprocessor {

    /**
     * @return a tag to be used in creating the name of the folder with the training result. This is necessary so we
     * can later compare different training results where the training was the same but the preprocessing was different.
     */
    String getTag();

    /**
     * Triggers the actual preprocessing.
     */
    void preprocess() throws Exception;


    /**
     * write preprocessing config to given folder
     * (necessary to be able to restore preprocessing for later testing for example)
     */
    void writeLog(final File folder) throws Exception;
}
