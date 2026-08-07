# Ten - MVP & Personas

## Personas

### 1. The Curator (primary persona)
Makes lists in a lot of different categories. Wants correlations to actually mean something, not just show up because an item is popular. For example, if almost everyone ranks "The Godfather," it'll "correlate" with tons of other popular movies just because so many people have it on their list, not because those movies are actually similar. The Curator would notice that and think it's a weak result. Since this persona is the one actually using the core feature of the app, what they'd consider a "good" correlation is what the correlation math needs to aim for.

### 2. The Browser
Mostly reads other people's public lists instead of posting their own. This persona is the reason a browse/feed screen is in scope for MVP - without one, there'd be no way to look at lists beyond your own or one person's profile at a time.

### 3. The Niche Obsessive (validation persona)
Deep in one narrow category, wants their deep-cut choices to correlate with something real rather than just share the general topic with others. This persona's job is to define the hardest test cases for entity resolution and embedding validation. If the app can correctly match this persona's obscure, deep-cut items, it'll handle everyone else's more common items too.

## MVP Definition

**Core loop:** post a list → the app computes and surfaces correlations involving your items → optionally drill into who (identifiable people only) shares that correlation. Separately, you can also browse other users' pages and public lists directly, not just through a correlation.