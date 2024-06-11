

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(craves b a)
(craves c b)
(craves d c)
(craves e d)
(province e)
)
(:goal
(and
(craves c b)
(craves d e)
(craves e c))
)
)


