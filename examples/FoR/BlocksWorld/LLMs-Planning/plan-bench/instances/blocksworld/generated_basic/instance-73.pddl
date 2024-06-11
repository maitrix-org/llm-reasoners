

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(ontable b)
(ontable c)
(on d c)
(clear a)
(clear b)
(clear d)
)
(:goal
(and
(on a d)
(on b c))
)
)


