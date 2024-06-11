

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b a)
(craves c d)
(craves d b)
(province c)
)
(:goal
(and
(craves b d)
(craves c a)
(craves d c))
)
)


