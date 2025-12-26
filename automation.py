"""
Default browser-use example using ChatBrowserUse

The simplest way to use browser-use - capable of any web task
with minimal configuration.
"""

import asyncio

from dotenv import load_dotenv

from browser_use import Agent, Browser, BrowserProfile, ChatBrowserUse

load_dotenv()


TASK_DESCRIPTION = """

Go on the website "https://atc-vertafore.inchannel.ai/zapier"

Then do the following:

Follow the given instructions to add an app to Creator Platform.

Steps to add:
How to Import App Actions into ATC. Follow the steps below to properly import actions from any app into ATC:
Search for the app in ATC and click Connect Authentication.

If you don’t already have an account for that app, create one using the  “Connect Authentication” button , then complete the OAuth authentication. Use the following email for all the app auths. aiagents@inchannelai.com. Make a new account when needed but with the same email. Use the logged in google account for aiagents@inchannelai.com wherever u can. 
After successful authentication, open the app and click on any action.
If the action’s input fields appear on the next screen, it confirms that authentication was successful.
Click the Import to ATC button.
This may take some time as the import runs in the background.
In the background, ATC will take the action ID and input schema, then convert them into an API endpoint format using the fields defined in ATC’s API Management.

Once the process is completed,the app shows as “Imported”
 
Start with app:
Google Contacts
Claude
Google Slides

NOTE: If any of the APP requires payment/ billing or for any reason you’re not able to add an app. Write that app name, reason for not adding, optional link (to redirect to billing, etc.) to this sheet: https://docs.google.com/spreadsheets/d/1N3esdkQpWPDjhJl25-bUcdhfJXvHiNIRxrGxsWLYUkQ/edit?usp=sharing
Sheet is already open on new tab.

"""



async def main():
    browser = Browser(
        use_cloud=False,
        executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        user_data_dir='~/Library/Application Support/Google/Chrome',
        profile_directory='Profile 40',
        headless=False,  # Show browser window
    )
    
    llm = ChatBrowserUse()
    agent = Agent(
        browser=browser,
        browser_profile=BrowserProfile(profile_directory='Profile 40'),
        task=TASK_DESCRIPTION,
        llm=llm,
    )
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())