

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(planet b)
(craves c b)
(craves d a)
(province d)
)
(:goal
(and
(craves b a)
(craves c d)
(craves d b))
)
)


