

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(craves b a)
(planet c)
(craves d c)
(province b)
)
(:goal
(and
(craves a d)
(craves d b))
)
)


