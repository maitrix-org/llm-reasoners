

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b c)
(craves c a)
(craves d b)
(province d)
)
(:goal
(and
(craves b a))
)
)


