package divisio;

import org.deeplearning4j.ui.api.UIServer;

/**
 * Starts the DL4J UI Server
 */
public class RunDL4JUI {
    public static void main( String[] args ) {
        UIServer uiServer = UIServer.getInstance();
        //Necessary: remote support is not enabled by default
        uiServer.enableRemoteListener();
    }
}
