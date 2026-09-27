"""
Legacy tool declarations bridge for Orthos.

Provides the original TOOL_DECLARATIONS list (38 tools) from main.py.original,
along with helper functions and a registry bridge for the new ToolRegistry.

Also provides TOOL_SUMMARIES — 1-line descriptions for dynamic tool discovery.
"""

TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": (
            "Opens or launches any application, website, or program on the computer. "
            "ALWAYS use this when the user says: open, launch, start, run, pull up, "
            "or 'open X real quick'. Examples: 'open WhatsApp', 'open Chrome', "
            "'launch Spotify', 'open calculator', 'pull up WhatsApp'. "
            "Do NOT use send_message just because the app is a messaging app — "
            "if the user only says to open it, call open_app."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {"type": "STRING", "description": "Name of the application or website to open"}
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "web_search",
        "description": (
            "Searches the web for any information. Use for: research, looking up "
            "current events, finding documentation, comparing products. "
            "Supports different search depths: fast (quick, no summarization), "
            "auto (balanced, default), deep (comprehensive with LLM summary)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":      {"type": "STRING", "description": "Search query"},
                "mode":       {"type": "STRING", "description": "search (default) or compare"},
                "items":      {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Items to compare"},
                "aspect":     {"type": "STRING", "description": "price | specs | reviews"},
                "type":       {"type": "STRING", "description": "Search depth: fast (quick, 3 results, no summary) | auto (balanced, 8 results, summarized) | deep (comprehensive, 20 results with detailed summary). Default: auto"},
                "numResults": {"type": "INTEGER", "description": "Explicit number of results to return (1-50). Overrides type-based count."},
            },
            "required": ["query"]
        }
    },
    {
        "name": "webfetch",
        "description": (
            "Fetches and reads the content of a URL the user GAVE you. "
            "Returns the raw page content. "
            "For deep research: use web_search(type='deep') first to discover info, "
            "then optionally webfetch to read specific pages. "
            "Do NOT use webfetch alone for research - the output bypasses LLM analysis."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "url": {"type": "STRING", "description": "The full URL to fetch and read, e.g. https://example.com/page"},
            },
            "required": ["url"]
        }
    },
    {
        "name": "weather_report",
        "description": "Gives the weather report to user",
        "parameters": {
            "type": "OBJECT",
            "properties": {"city": {"type": "STRING", "description": "City name"}},
            "required": ["city"]
        }
    },
    {
        "name": "send_message",
        "description": (
            "Sends a message to a specific person via WhatsApp, Telegram, or similar. "
            "ONLY use this when the user explicitly provides BOTH a recipient AND message content. "
            "Example triggers: 'text John saying I am late', 'send a WhatsApp to mom that dinner is ready'. "
            "Do NOT call this if the user only wants to open the app without sending a message."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "receiver":     {"type": "STRING", "description": "Recipient contact name"},
                "message_text": {"type": "STRING", "description": "The exact message text to send"},
                "platform":     {"type": "STRING", "description": "Platform: WhatsApp, Telegram, etc."}
            },
            "required": ["receiver", "message_text", "platform"]
        }
    },
    {
        "name": "reminder",
        "description": "Sets a timed reminder using Task Scheduler.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date":    {"type": "STRING", "description": "Date in YYYY-MM-DD format"},
                "time":    {"type": "STRING", "description": "Time in HH:MM format (24h)"},
                "message": {"type": "STRING", "description": "Reminder message text"}
            },
            "required": ["date", "time", "message"]
        }
    },
    {
        "name": "youtube_video",
        "description": (
            "Controls YouTube. Use for: playing videos, summarizing a video's content, "
            "getting video info, or showing trending videos."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play | summarize | get_info | trending"},
                "query":  {"type": "STRING", "description": "Search query for play action"},
                "save":   {"type": "BOOLEAN", "description": "Save summary to Notepad"},
                "region": {"type": "STRING", "description": "Country code for trending e.g. TR, US"},
                "url":    {"type": "STRING", "description": "Video URL for get_info action"},
            },
            "required": []
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures screen/camera and returns the image directly to you. "
            "You analyze it yourself using your built-in vision. "
            "Call this for ANY visual question: cursor location, screen "
            "content, UI analysis, camera input, zoomed-in verification."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle":    {"type": "STRING", "description": "'screen' or 'camera'. Default: 'screen'"},
                "text":     {"type": "STRING", "description": "The question about the captured image"},
                "verify_x": {"type": "INTEGER", "description": "X coordinate for cursor marker on screenshot (optional)"},
                "verify_y": {"type": "INTEGER", "description": "Y coordinate for cursor marker on screenshot (optional)"},
                "zoom":     {"type": "INTEGER", "description": "Crop radius in pixels for verification close-ups. Smaller = more zoom. Default 350. Use 200-600."},
            },
            "required": ["text"]
        }
    },
    {
        "name": "screen_locate",
        "description": (
            "Finds ANY visual element on screen by description (~1-2 seconds). "
            "Uses Moondream point() API — finds buttons, icons, input fields, "
            "text, sliders, images. Works on any element, not just text. "
            "Returns coordinates like '\"element\" at pixel (x, y)'. "
            "Use MCP WinApp server (mcp_winapp_) for native Windows app automation."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "text": {"type": "STRING", "description": "Visual description of the element to locate, e.g. 'blue login button', 'video titled MODI ON MUTE', 'search input field with magnifying glass icon'"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "computer_settings",
        "description": (
            "Controls the computer: volume, brightness, window management, keyboard shortcuts, "
            "typing text on screen, closing apps, fullscreen, dark mode, WiFi, restart, shutdown, "
            "scrolling, tab management, zoom, screenshots, lock screen, refresh/reload page."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "The action to perform"},
                "description": {"type": "STRING", "description": "Natural language description"},
                "value":       {"type": "STRING", "description": "Optional value"}
            },
            "required": []
        }
    },
    {
        "name": "file_controller",
        "description": "Manages files and folders: list, create, delete, move, copy, rename, read, write, find, disk usage.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "list | create_file | create_folder | delete | move | copy | rename | read | write | find | largest | disk_usage | organize_desktop | info"},
                "path":        {"type": "STRING", "description": "File/folder path or shortcut: desktop, downloads, documents, home"},
                "destination": {"type": "STRING", "description": "Destination path for move/copy"},
                "new_name":    {"type": "STRING", "description": "New name for rename"},
                "content":     {"type": "STRING", "description": "Content for create_file/write"},
                "name":        {"type": "STRING", "description": "File name to search for"},
                "extension":   {"type": "STRING", "description": "File extension to search"},
                "count":       {"type": "INTEGER", "description": "Number of results for largest"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "desktop_control",
        "description": "Controls the desktop: wallpaper, organize, clean, list, stats.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "wallpaper | wallpaper_url | organize | clean | list | stats | task"},
                "path":   {"type": "STRING", "description": "Image path for wallpaper"},
                "url":    {"type": "STRING", "description": "Image URL for wallpaper_url"},
                "mode":   {"type": "STRING", "description": "by_type or by_date for organize"},
                "task":   {"type": "STRING", "description": "Natural language desktop task"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "code_helper",
        "description": "Writes, edits, explains, runs, or builds code files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "write | edit | explain | run | build | auto"},
                "description": {"type": "STRING", "description": "What the code should do"},
                "language":    {"type": "STRING", "description": "Programming language"},
                "output_path": {"type": "STRING", "description": "Where to save the file"},
                "file_path":   {"type": "STRING", "description": "Path to existing file"},
                "code":        {"type": "STRING", "description": "Raw code string for explain"},
                "args":        {"type": "STRING", "description": "CLI arguments"},
                "timeout":     {"type": "INTEGER", "description": "Execution timeout in seconds"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "dev_agent",
        "description": "Builds complete multi-file projects from scratch.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "description":  {"type": "STRING", "description": "What the project should do"},
                "language":     {"type": "STRING", "description": "Programming language"},
                "project_name": {"type": "STRING", "description": "Optional project folder name"},
                "timeout":      {"type": "INTEGER", "description": "Run timeout in seconds"},
            },
            "required": ["description"]
        }
    },
    {
        "name": "agent_task",
        "description": (
            "Executes complex multi-step tasks requiring multiple different tools. "
            "Examples: 'research X and save to file', 'find and organize files'. "
            "DO NOT use for single commands."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "goal":     {"type": "STRING", "description": "Complete description of what to accomplish"},
                "priority": {"type": "STRING", "description": "low | normal | high"}
            },
            "required": ["goal"]
        }
    },
    {
        "name": "computer_control",
        "description": "Direct computer control: type, click, hotkeys, scroll, move mouse, screenshots, find elements on screen, get cursor position, verified move with position check.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "type | smart_type | click | double_click | right_click | hotkey | press | scroll | move | drag | copy | paste | screenshot | wait | clear_field | focus_window | screen_find | screen_click | random_data | user_data"},
                "text":        {"type": "STRING", "description": "Text to type or paste"},
                "x":           {"type": "INTEGER", "description": "X coordinate"},
                "y":           {"type": "INTEGER", "description": "Y coordinate"},
                "keys":        {"type": "STRING", "description": "Key combination e.g. 'ctrl+c'"},
                "key":         {"type": "STRING", "description": "Single key e.g. 'enter'"},
                "direction":   {"type": "STRING", "description": "up | down | left | right"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount"},
                "seconds":     {"type": "NUMBER",  "description": "Seconds to wait"},
                "title":       {"type": "STRING",  "description": "Window title for focus_window"},
                "description": {"type": "STRING",  "description": "Element description"},
                "type":        {"type": "STRING",  "description": "Data type for random_data"},
                "field":       {"type": "STRING",  "description": "Field for user_data"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing"},
                "path":        {"type": "STRING",  "description": "Save path for screenshot"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "game_updater",
        "description": (
            "THE ONLY tool for ANY Steam or Epic Games request. "
            "Use for: installing, downloading, updating games, listing installed games."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING",  "description": "update | install | list | download_status | schedule | cancel_schedule | schedule_status"},
                "platform":  {"type": "STRING",  "description": "steam | epic | both"},
                "game_name": {"type": "STRING",  "description": "Game name"},
                "app_id":    {"type": "STRING",  "description": "Steam AppID"},
                "hour":      {"type": "INTEGER", "description": "Hour for scheduled update 0-23"},
                "minute":    {"type": "INTEGER", "description": "Minute for scheduled update 0-59"},
                "shutdown_when_done": {"type": "BOOLEAN", "description": "Shut down PC when done"},
            },
            "required": []
        }
    },
    {
        "name": "flight_finder",
        "description": "Searches Google Flights and speaks the best options.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "origin":      {"type": "STRING",  "description": "Departure city or airport code"},
                "destination": {"type": "STRING",  "description": "Arrival city or airport code"},
                "date":        {"type": "STRING",  "description": "Departure date"},
                "return_date": {"type": "STRING",  "description": "Return date for round trips"},
                "passengers":  {"type": "INTEGER", "description": "Number of passengers"},
                "cabin":       {"type": "STRING",  "description": "economy | premium | business | first"},
                "save":        {"type": "BOOLEAN", "description": "Save results to Notepad"},
            },
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "run_terminal",
        "description": (
            "Executes a shell/terminal command and returns the output. "
            "Use for: system info (systeminfo), processes (tasklist), "
            "network (ipconfig, ping, netstat), file operations (dir, find), "
            "or any CLI tool. The command runs via cmd.exe on Windows or bash on Linux/Mac. "
            "Always explain what command you're about to run before executing it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "command": {"type": "STRING", "description": "The exact command to execute, e.g. 'tasklist', 'ipconfig /all', 'dir C:\\'"},
                "timeout": {"type": "INTEGER", "description": "Timeout in seconds (default 30, max 120)"},
                "cwd":     {"type": "STRING", "description": "Working directory (optional, defaults to project root)"},
            },
            "required": ["command"]
        }
    },
    {
        "name": "list_mcp_servers",
        "description": (
            "Lists all configured MCP servers with their status (enabled/disabled/running/error), "
            "tool counts, descriptions, and how to use their tools. "
            "All MCP tools follow the naming pattern: mcp_{server_name}_{tool_name}. "
            "Use this when you need to discover what MCP servers/tools are available."
        ),
        "parameters": {"type": "OBJECT", "properties": {}}
    },
    {
        "name": "manage_mcp_server",
        "description": (
            "Manage Orthos MCP servers on the user's request. Use this for requests such as "
            "'start GitHub MCP', 'stop Playwright', 'restart filesystem MCP', or "
            "'install package-name MCP'. Always list servers first if the requested name is unclear."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "start, stop, restart, enable, disable, or install"},
                "server": {"type": "STRING", "description": "Configured server name; required except for install"},
                "package": {"type": "STRING", "description": "npm or uvx package for install, e.g. @modelcontextprotocol/server-github"},
                "command": {"type": "STRING", "description": "Optional install runtime: npx or uvx (default npx)"},
                "category": {"type": "STRING", "description": "Optional category for an installed server"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "shutdown_orthos",
        "description": (
            "Shuts down the assistant completely. "
            "Call this when the user expresses intent to end the conversation, "
            "close the assistant, say goodbye, or stop Orthos."
        ),
        "parameters": {"type": "OBJECT", "properties": {}}
    },
    {
        "name": "file_processor",
        "description": (
            "Processes any file that the user has uploaded or dropped onto the interface. "
            "Supports: images, PDFs, Word docs, CSV/Excel, JSON, code files, audio, video, archives."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path":   {"type": "STRING",  "description": "Full path to the uploaded file"},
                "action":      {"type": "STRING",  "description": "What to do with the file"},
                "instruction": {"type": "STRING",  "description": "Free-form instruction"},
                "format":      {"type": "STRING",  "description": "Target format for conversion"},
                "width":       {"type": "INTEGER", "description": "Target width for image resize"},
                "height":      {"type": "INTEGER", "description": "Target height for image resize"},
                "scale":       {"type": "NUMBER",  "description": "Scale factor"},
                "quality":     {"type": "INTEGER", "description": "Quality 1-100"},
                "start":       {"type": "STRING",  "description": "Start time for trim"},
                "end":         {"type": "STRING",  "description": "End time for trim"},
                "timestamp":   {"type": "STRING",  "description": "Timestamp for video frame extraction"},
                "column":      {"type": "STRING",  "description": "Column name for CSV filter/sort"},
                "value":       {"type": "STRING",  "description": "Filter value"},
                "condition":   {"type": "STRING",  "description": "Filter condition"},
                "ascending":   {"type": "BOOLEAN", "description": "Sort order"},
                "save":        {"type": "BOOLEAN", "description": "Save result to file"},
                "destination": {"type": "STRING",  "description": "Output folder for archive extract"},
            },
            "required": []
        }
    },
    {
        "name": "save_memory",
        "description": (
            "Save ANY information to permanent long-term memory. "
            "Call this silently whenever you learn something about the user: name, age, location, job, "
            "preferences, ongoing projects, goals, relationships, habits, past conversations, "
            "mistakes, lessons — literally anything worth remembering across sessions. "
            "Better to save too much than too little. Never announce you are saving."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity | preferences | projects | relationships | wishes | procedures | notes"
                    )
                },
                "key":   {"type": "STRING", "description": "Short snake_case key, e.g. 'name', 'favorite_color', 'current_project'"},
                "value": {"type": "STRING", "description": "The value to remember"},
            },
            "required": ["category", "key", "value"]
        }
    },
    {
        "name": "forget_memory",
        "description": "Delete specific information from long-term memory. Only call this when the user explicitly asks you to forget or delete something.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "key":      {"type": "STRING", "description": "The key of the memory to delete"},
                "category": {"type": "STRING", "description": "Category: identity | preferences | projects | relationships | wishes | notes. Default: notes"},
            },
            "required": ["key"]
        }
    },
    {
        "name": "search_memory",
        "description": (
            "CRITICAL: SEARCH MEMORY BEFORE ANSWERING when you need ANY context.\n\n"
            "You have NO long-term facts in your system prompt anymore except CORE MEMORY. "
            "Your entire memory system works through TOOLS:\n"
            "  - search_memory(query)  → multi-signal search (keyword + semantic + entity + identity + importance)\n"
            "  - recall_conversation(query)  → search RAW past chat messages VERBATIM (exact quotes, dates, session names)\n"
            "  - save_memory(cat, key, value)  → store ANY fact permanently\n"
            "  - forget_memory(key)  → delete a fact\n"
            "  - save_procedure(key, value)  → store a learned workflow or 'how-to' rule\n"
            "  - list_memories()  → see everything I know about you\n"
            "  - list_procedures()  → see all learned workflows and rules\n"
            "  - core_memory_append(key, value)  → add to always-in-context identity facts\n"
            "  - core_memory_replace(key, value)  → update a core identity fact\n"
            "  - archival_memory_search(query)  → search archived (evicted) memories\n"
            "  - context_status()  → check memory health, usage, and indexes\n\n"
            "TRAINING RULE: If a user asks about ANYTHING that requires recalling past "
            "facts, preferences, projects, names, or history — "
            "you MUST call search_memory FIRST before answering. "
            "If the user asks WHAT WAS SAID in a past conversation ('yaad hai', 'we talked about', "
            "'what did I say about', 'kya baat ki thi', quoting an old message) — call "
            "recall_conversation FIRST: it scans every raw chat message verbatim, while "
            "search_memory only knows distilled facts.\n\n"
            "This search uses 5 signals:\n"
            "  1. KEYWORD:  BM25- ranked keyword search with query expansion (synonyms)\n"
            "  2. SEMANTIC: vector similarity with MiniLM (understands meaning)\n"
            "  3. ENTITY:   linked facts across categories via entity graph\n"
            "  4. IDENTITY: auto-returns all known facts if you ask 'who am I' or similar\n"
            "  5. IMPORTANCE: score-weighted facts boost highly-relevant entries\n"
            "  (+ raw conversation matches surface automatically when relevant)"
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Search term or phrase — short keywords or full questions. Try searching for names, topics, or even feelings!"},
                "limit": {"type": "INTEGER", "description": "Max results to return (default 10)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "recall_conversation",
        "description": (
            "Search RAW PAST CONVERSATIONS verbatim. Use this whenever the user wants to "
            "recall what was actually SAID in an earlier chat — e.g. 'yaad hai humne kya baat ki thi', "
            "'what did I tell you about X', 'humne us topic pe kya discuss kiya tha', or when they "
            "quote/refer to an old message. Returns the exact matched messages with date, "
            "session name, and speaker. Unlike search_memory (distilled facts only), this scans "
            "every raw chat message ever saved — keyword AND meaning based. "
            "Set full_session=true to expand a matched message's whole session into context."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "What to find in past conversations — keywords, a phrase, or a description of the topic. Hinglish works too."},
                "limit": {"type": "INTEGER", "description": "Max matched messages to return (default 8)"},
                "full_session": {"type": "BOOLEAN", "description": "If true, also return the full session of the best match (use sparingly — long output)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_memories",
        "description": "Show a complete summary of everything I know about the user across ALL categories (identity, preferences, projects, relationships, wishes, procedures, notes). Call this when the user asks 'what do you know about me' or 'show me my memories'. This also clears expired memories automatically.",
        "parameters": {
            "type": "OBJECT",
            "properties": {}
        }
    },
    {
        "name": "save_procedure",
        "description": (
            "Store a learned workflow, rule, pattern, or 'how-to' knowledge. "
            "Use this when you learn something about HOW to do things — "
            "for example: a specific tool workflow, a coding convention, "
            "a preferred sequence of steps, a response format rule, "
            "or any procedural knowledge that helps you do your job better.\n\n"
            "Examples:\n"
            "  - 'always verify after clicking'\n"
            "  - 'use Alt+F4 to close windows'\n"
            "  - 'wait 3 seconds before verification'\n"
            "  - 'never use emojis in responses'\n\n"
            "This is different from save_memory — that stores FACTS about the user.\n"
            "save_procedure stores SKILLS and WORKFLOWS."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "key":   {"type": "STRING", "description": "Short snake_case name for the procedure, e.g. 'click_verification_workflow', 'window_close_method', 'response_format_rule'"},
                "value": {"type": "STRING", "description": "The detailed procedure steps or rule"},
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "list_procedures",
        "description": "Show all learned procedures, workflows, and behavioral rules. Call this when starting a new task to remember HOW to do things properly. Returns every stored procedure with key and value.",
        "parameters": {
            "type": "OBJECT",
            "properties": {}
        }
    },
    {
        "name": "core_memory_append",
        "description": "Append a NEW fact to core memory (identity). Use ONLY for permanent facts about the user that are always relevant (name, occupation, location, core preferences). Cannot overwrite — use core_memory_replace if key exists.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "key":   {"type": "STRING", "description": "snake_case key, e.g. 'name', 'occupation', 'location'"},
                "value": {"type": "STRING", "description": "The fact value"}
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "core_memory_replace",
        "description": "Replace an EXISTING fact in core memory (identity). Updates the value for a key that already exists. Use when the user corrects or updates a core fact.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "key":   {"type": "STRING", "description": "Existing key to update"},
                "value": {"type": "STRING", "description": "New value"}
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "archival_memory_search",
        "description": "Search ARCHIVED memory (facts that were evicted from main memory due to capacity limits). Use this when search_memory returns nothing — the data may be in the archive.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":  {"type": "STRING", "description": "Search term"},
                "limit":  {"type": "INTEGER", "description": "Max results (default 10)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "context_status",
        "description": "Show memory usage, token estimates, index health, and per-category breakdown. Call this to check if memory is healthy or if you need to manage space.",
        "parameters": {
            "type": "OBJECT",
            "properties": {}
        }
    },
    {
        "name": "search_timeline",
        "description": (
            "Search the project timeline for historical events and milestones. "
            "Use this when asked about past events, history, 'what happened', "
            "'when did we', or any question about project chronology. "
            "Supports semantic search (natural language queries) and temporal filters."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":      {"type": "STRING", "description": "Search query or event description"},
                "limit":      {"type": "INTEGER", "description": "Max results (default 10)"},
                "event_type": {"type": "STRING", "description": "Filter: integration | installation | migration | creation | fix | deployment | benchmark | note | reflection"},
                "days_back":  {"type": "INTEGER", "description": "Only events within this many days (0 = no filter)"},
            },
            "required": ["query"]
        }
    },
    {
        "name": "save_timeline_event",
        "description": (
            "Manually log an important event to the timeline. "
            "Use when something notable happens that should be remembered chronologically: "
            "project milestones, decisions, deployments, discoveries, or achievements. "
            "High-importance events (>=0.8) are marked with ★ in timeline display."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "event_type":  {"type": "STRING", "description": "integration | installation | migration | creation | fix | deployment | benchmark | note | reflection"},
                "description": {"type": "STRING", "description": "What happened — be specific and descriptive"},
                "importance":  {"type": "NUMBER", "description": "0.0 to 1.0. Use 0.8+ for major milestones"},
            },
            "required": ["event_type", "description"]
        }
    },
    {
        "name": "list_projects",
        "description": "List all known projects. Use this to discover what projects exist and their IDs.",
        "parameters": {
            "type": "OBJECT",
            "properties": {}
        }
    },
    {
        "name": "create_project",
        "description": "Create a new project for organizing memories, sessions, and timelines. Use when starting a new major initiative.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING", "description": "Project name, e.g. 'Shopify Store', 'Orthos Assistant'"},
            },
            "required": ["name"]
        }
    },
    {
        "name": "set_active_project",
        "description": "Switch the current session to a different project. All subsequent memories and timeline events will be scoped to this project.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "project_id": {"type": "STRING", "description": "Project ID from list_projects or project name"},
            },
            "required": ["project_id"]
        }
    },
    {
        "name": "get_project_memories",
        "description": "Get all memories and context for a specific project. Shows project-scoped facts, recent sessions, and timeline.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "project_id": {"type": "STRING", "description": "Project ID or name"},
            },
            "required": ["project_id"]
        }
    },
    {
        "name": "save_tool_state",
        "description": (
            "Save the current state of a tool for continuity across calls. "
            "Use when you want to remember what a tool was doing. "
            "For example: browser URL, Python working directory, current file being edited."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "tool_name": {"type": "STRING", "description": "Tool name: browser | python | vision | git | terminal | file_controller | code_helper"},
                "key":       {"type": "STRING", "description": "State key, e.g. 'current_url', 'last_cwd', 'last_file'"},
                "value":     {"type": "STRING", "description": "The state value"},
            },
            "required": ["tool_name", "key", "value"]
        }
    },
    {
        "name": "get_tool_state",
        "description": "Get the remembered state for a tool. Use this at the start of a multi-step task to pick up where you left off.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "tool_name": {"type": "STRING", "description": "browser | python | vision | git | terminal | file_controller | code_helper"},
            },
            "required": ["tool_name"]
        }
    },
    {
        "name": "search_knowledge_graph",
        "description": (
            "Search the knowledge graph for entities and their relationships. "
            "Shows direct connections AND multi-hop relationships between concepts. "
            "Use when you need to understand HOW things are connected, not just find similar text. "
            "For example: 'Which database stores Shopify data?' → follows Shopify → Project → Memory → Chroma chain."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":    {"type": "STRING", "description": "Entity name or topic to explore in the graph"},
                "max_hops": {"type": "INTEGER", "description": "How many relationship steps to traverse (default 2, max 3)"},
            },
            "required": ["query"]
        }
    },
    {
        "name": "read",
        "description": (
            "Read the contents of a file from the local filesystem. "
            "Supports reading entire files or specific line ranges for large files. "
            "Use when you need to see what's inside a file or read specific sections. "
            "Returns the file content with metadata about line count."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path": {"type": "STRING", "description": "Full path to the file to read"},
                "offset": {"type": "INTEGER", "description": "Line number to start reading from (1-indexed). If None, starts from line 1"},
                "limit": {"type": "INTEGER", "description": "Maximum number of lines to read. If None, reads until end of file"},
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "edit",
        "description": (
            "Modify a file by replacing exact text with new text. "
            "Uses precise string replacement - provide the exact old_string to find and the new_string to replace it with. "
            "Supports single replacement or replace_all for multiple occurrences. "
            "Use for: fixing bugs, updating content, refactoring code, adding comments. "
            "IMPORTANT: The old_string must match exactly including whitespace and newlines."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path": {"type": "STRING", "description": "Full path to the file to edit"},
                "old_string": {"type": "STRING", "description": "Exact text to find and replace. Must match including whitespace and newlines."},
                "new_string": {"type": "STRING", "description": "Text to replace the old_string with"},
                "replace_all": {"type": "BOOLEAN", "description": "If true, replace all occurrences. If false, replace only the first one. Default: false"},
            },
            "required": ["file_path", "old_string", "new_string"]
        }
    },
    {
        "name": "grep",
        "description": (
            "Search file contents using regex patterns. "
            "Returns matching lines with file paths and line numbers. "
            "Use when you need to find specific code, text, or patterns across multiple files. "
            "Supports filtering by file pattern (include), exclusion patterns, and .gitignore compliance. "
            "Example: find all Python files containing 'TODO', exclude test files."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "pattern": {"type": "STRING", "description": "Regex pattern to search for"},
                "path": {"type": "STRING", "description": "Directory to search in. If None, searches the Orthos project folder"},
                "include": {"type": "STRING", "description": "File pattern to include, e.g., '*.py', '*.js'. If None, searches all files."},
                "exclude": {"type": "STRING", "description": "Patterns to exclude, separated by |, e.g., '__pycache__|*.pyc|node_modules'"},
                "max_results": {"type": "INTEGER", "description": "Maximum number of results to return. Default: 100"},
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "glob",
        "description": (
            "Find files matching a glob pattern. "
            "Use when you need to discover files of a certain type, e.g., all Python files, all JSON configs. "
            "Supports recursive patterns like '**/*.py' for deep searching. "
            "Results are sorted by modification time (most recent first). "
            "Example: '**/*.py' finds all Python files, 'src/**/*.ts' finds TypeScript in src."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "pattern": {"type": "STRING", "description": "Glob pattern, e.g., '**/*.py', '*.js', 'src/**/*.ts'"},
                "path": {"type": "STRING", "description": "Directory to search in. If None, searches the Orthos project folder"},
            },
            "required": ["pattern"]
        }
    },
]


# ---------------------------------------------------------------------------
# Type conversion helpers
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "OBJECT": "object", "STRING": "string", "ARRAY": "array",
    "INTEGER": "integer", "BOOLEAN": "boolean", "NUMBER": "number",
}


def _convert_type(t: str) -> str:
    return _TYPE_MAP.get(t, t.lower()) if isinstance(t, str) else t


def _convert_props(props: dict) -> dict:
    out = {}
    for k, v in props.items():
        nv = dict(v)
        if "type" in nv:
            nv["type"] = _convert_type(nv["type"])
        if "items" in nv and isinstance(nv["items"], dict):
            nv["items"] = {"type": _convert_type(nv["items"].get("type", "string"))}
        out[k] = nv
    return out


def _to_ollama_tools(decls: list) -> list:
    tools = []
    for d in decls:
        params = d.get("parameters", {})
        new_params: dict = {
            "type":       "object",
            "properties": _convert_props(params.get("properties", {})),
        }
        req = params.get("required")
        if req:
            new_params["required"] = req
        tools.append({
            "type": "function",
            "function": {
                "name":        d["name"],
                "description": d["description"],
                "parameters":  new_params,
            },
        })
    return tools


OLLAMA_TOOLS = _to_ollama_tools(TOOL_DECLARATIONS)


# ---------------------------------------------------------------------------
# Registry bridge — creates wrappers that call the right action function
# ---------------------------------------------------------------------------

_ACTION_MODULE_MAP: dict[str, tuple[str, str]] = {
    "open_app":         ("actions.open_app", "open_app"),
    "web_search":       ("actions.web_search", "web_search"),
    "webfetch":         ("actions.web_search", "webfetch"),
    "weather_report":   ("actions.weather_report", "weather_action"),
    "send_message":     ("actions.send_message", "send_message"),
    "reminder":         ("actions.reminder", "reminder"),
    "youtube_video":    ("actions.youtube_video", "youtube_video"),
    "screen_process":   ("actions.screen_processor", "screen_process"),
    "screen_locate":    ("actions.screen_processor", "screen_locate"),
    "computer_settings":("actions.computer_settings", "computer_settings"),
    "file_controller":  ("actions.file_controller", "file_controller"),
    "desktop_control":  ("actions.desktop", "desktop_control"),
    "code_helper":      ("actions.code_helper", "code_helper"),
    "dev_agent":        ("actions.dev_agent", "dev_agent"),
    "computer_control": ("actions.computer_control", "computer_control"),
    "game_updater":     ("actions.game_updater", "game_updater"),
    "flight_finder":    ("actions.flight_finder", "flight_finder"),
    "run_terminal":     ("actions.terminal", "run_terminal"),
    "file_processor":   ("actions.file_processor", "file_processor"),
    # Tools handled inline (memory, agent, etc.) — use a no-op wrapper
    "save_memory":           (None, None),
    "forget_memory":         (None, None),
    "search_memory":         (None, None),
    "recall_conversation":   (None, None),
    "list_memories":         (None, None),
    "save_procedure":        (None, None),
    "list_procedures":       (None, None),
    "core_memory_append":    (None, None),
    "core_memory_replace":   (None, None),
    "archival_memory_search":(None, None),
    "context_status":        (None, None),
    "agent_task":            (None, None),
    "list_mcp_servers":      (None, None),
    "manage_mcp_server":     (None, None),
    "shutdown_orthos":       (None, None),
    # Timeline tools
    "search_timeline":       (None, None),
    "save_timeline_event":   (None, None),
    # Project tools
    "list_projects":         (None, None),
    "create_project":        (None, None),
    "set_active_project":    (None, None),
    "get_project_memories":  (None, None),
    # Tool memory
    "save_tool_state":       (None, None),
    "get_tool_state":        (None, None),
    # GraphRAG
    "search_knowledge_graph":(None, None),
    # File tools
    "read":               ("actions.read", "read_tool"),
    "edit":               ("actions.edit", "edit_tool"),
    "grep":               ("actions.grep", "grep_tool"),
    "glob":               ("actions.glob", "glob_tool"),
}


def _make_legacy_handler(name: str):
    """Create a wrapper function that dispatches to the correct action module."""
    mapping = _ACTION_MODULE_MAP.get(name)

    if mapping and mapping[0] is not None:
        module_path, func_name = mapping

        def handler(parameters=None, response=None, player=None, **kwargs):
            try:
                import importlib
                mod = importlib.import_module(module_path)
                func = getattr(mod, func_name)
                return func(parameters=parameters, response=response, player=player, **kwargs)
            except Exception as e:
                return f"Error executing {name}: {e}"
    else:
        # Inline-handled tools — no-op wrapper (actual execution is in _exec_tool_inner)
        def handler(parameters=None, response=None, player=None, **kwargs):
            return "Done."

    handler.__name__ = name
    return handler


def register_legacy_tools(registry):
    """Register all legacy TOOL_DECLARATIONS into a ToolRegistry instance.

    Args:
        registry: A ToolRegistry instance.

    Returns:
        The same registry, populated with legacy tools.
    """
    for decl in TOOL_DECLARATIONS:
        name = decl["name"]
        handler = _make_legacy_handler(name)
        registry.register(
            func=handler,
            name=name,
            description=decl["description"],
            parameters=decl.get("parameters", {}),
        )
    return registry


from core.tool_index import TOOL_SUMMARIES, META_TOOL_DEFINITIONS
